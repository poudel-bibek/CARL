"""
Controllers written for Density Aware RL agent
IDM controllers that can also provide shock

This Modified IDM Controller contains: 

**This does not come from any past published works with other forms of modifications** 
"""
import math 
import numpy as np
from flow.controllers.base_controller import BaseController

import torch
import torch.nn as nn

class ModifiedIDMController(BaseController):
    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True,
                 car_following_params=None,

                 shock_vehicle = False,):
        """Instantiate an IDM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

        self.shock_vehicle = shock_vehicle
        self.shock_acceleration = 0.0 # Default
        self.shock_time = False # Per time step decision on whether to shock or not

    def get_accel(self, env):
        """
        it will automatically call this for each vehicle
        At shock times, we have to return the shock acceleration
        """
        # If the vehicle is a registered shock vehicle and shock model says shock now
        if self.shock_vehicle and self.shock_time:
            return self.get_shock_accel()
        else: 
            return self.get_idm_accel(env)

    def set_shock_accel(self, accel):
        self.shock_acceleration = accel
        #print(f"\nFrom the controller: {self.veh_id, self.shock_acceleration}\n")
        #return accel
    
    def get_shock_accel(self):
        #print("Shock")
        return self.shock_acceleration

    def set_shock_time(self, shock_time):
        self.shock_time = shock_time

    def get_shock_time(self):
        return self.shock_time

    def get_idm_accel(self, env):
        """
        it will automatically call this for each vehicle
        At shock times, we have to return the shock acceleration
        """
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        #print("IDM")
        return self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)

class ImitationLearningController(BaseController):
    """

    Model 1: Apply imitation learning at all times
    Model 2: Apply imititation learning when asked for perturbations. Use model 2 now.

    Since car following accelerations dont contain scenerios:
        1. Where the vehicle has to start from stop.
        2. When there is no other vehicles in front. (In the ring at least there is always a leader)
        3. The 800 param model when run on ~300 vehicles peak in the Bottleneck will be a problem.

    Add stochasticity to the imitation actions?
    It is not able to produce extreme accelerations. 

    """
    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True,
                 car_following_params=None,
                 
                 # By default, the shock_vehicle is set to False. When is it turned ON?
                 shock_vehicle = False,
                 ):

        
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )

        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

        self.shock_vehicle = shock_vehicle
        self.shock_acceleration = 0.0 # Default
        self.shock_time = False # Per time step decision on whether to shock or not
        self.imitation_model = self.load_imitation_model()
        
    def load_imitation_model(self, ):
        """
        Load the Imitation Learning Model
        """

        # 800 parameter model
        class NeuralNet(nn.Module):
            def __init__(self, dropout_rate=0.15): # dropour probability 15%
                super(NeuralNet, self).__init__() 
                self.fc1 = nn.Linear(3, 32)
                self.relu = nn.ReLU()
                # Adding dropout after the first ReLU
                self.dropout1 = nn.Dropout(dropout_rate)
                self.fc2 = nn.Linear(32, 16)
                # Adding dropout after the second ReLU
                self.dropout2 = nn.Dropout(dropout_rate)
                
                self.fc3 = nn.Linear(16, 8)
                self.fc4 = nn.Linear(8, 1)

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                # out = self.dropout1(out) # Dropouts turned OFF at eval
                
                out = self.fc2(out)
                out = self.relu(out)
                # out = self.dropout2(out) # Dropouts turned OFF at eval
                
                out = self.fc3(out) 
                out = self.relu(out)
                out = self.fc4(out) # Last layer, no activation
                return out
        
        url = "https://huggingface.co/matrix-multiply/Imitation_Learning_EnduRL/resolve/main/imitation_best_model.pth?download=true"
        saved_best_net = NeuralNet()
        state_dict = torch.hub.load_state_dict_from_url(url)
        saved_best_net.load_state_dict(state_dict)
        saved_best_net.eval()
        return saved_best_net
                
    def get_idm_accel(self, env):
        """
        it will automatically call this for each vehicle
        At shock times, we have to return the shock acceleration
        """
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        #print("IDM")
        return self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)
        
    def get_shock_accel(self, ):
        """
        This is not even used.
        To make this controller compatible with rest of the system (where shock was sampled earlier) 
        """
        return self.shock_acceleration

    def set_shock_accel(self, accel, env=None):
        """
        Used to provide the pre-calculated acceleration to the RV.
        Just to make it compatible with the rest of the system.
        Although the acceleration `accel` is provided, it is not used.
        The acceleration is gotten from the imitation model. 
        """
        if env is not None:
            imitation_accel = self.get_imitation_accel(env)
            self.shock_acceleration = imitation_accel
        else: 
            print("\nError: Environment not provided to the imitation model\n")
        

    def set_shock_time(self, shock_time):
        """
        If we follow model 2: i.e., provide imitation model shocks on pre-calculated shock times.
        """
        self.shock_time = shock_time

    def get_shock_time(self):
        """
        This is not even used.
        """
        return self.shock_time

    
    def get_imitation_accel(self, env):
        """
        Takes 3 inputs in this order 'headway', 'ego_velocity', 'leader_velocity'
        Make inference on the model and return the acceleration

        One single acceleration is gotten every timestep during the shock duration.
        """
        lead_id = env.k.vehicle.get_leader(self.veh_id)

        # Get the inputs
        headway = env.k.vehicle.get_headway(self.veh_id)
        ego_vel = env.k.vehicle.get_speed(self.veh_id)
        
        # In the ring at-least there will always be a leader
        if lead_id is None or lead_id == '':  # no car ahead
            lead_vel = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
        print(f"Lead id: {lead_id}, Headway: {headway}, Vel: {lead_vel}; ego Vel: {ego_vel}")

        # Normalize the inputs (normalizers similar to training)
        max_vel = 70
        max_headway = 150
        
        headway = headway / max_headway
        ego_vel = ego_vel / max_vel
        lead_vel = lead_vel / max_vel

        # Pass the inputs to the model
        inputs = torch.tensor([headway, ego_vel, lead_vel], dtype=torch.float32)
        imitation_accel = self.imitation_model(inputs).item()
        print(f"Imitation Acceleration: {imitation_accel}\n")
        

        # Decide based on headway of the ego vehicle which acceleration to return (sampled vs imitation)



        # Add some stochasticity to the imitation actions but keep within the range
        # First sample acceeration uniformly outside the nominal [-1, 1] range
        imitation_accel = np.clip(imitation_accel + np.random.uniform(-3, 3), -3, 3)

        return imitation_accel

    def get_accel(self, env):
        """
        Following the model 2: manual implementation of the vehicle switch instead of TRACI
        """
        # If the vehicle is a registered shock vehicle and shock model says shock now
        if self.shock_vehicle and self.shock_time:
            return self.get_shock_accel()
        else: 
            return self.get_idm_accel(env)



class TrainedAgentController(BaseController):
    """
    For efficiency leader, the action space also need free flow estimation and action =0.0 at speed greater than that.
    Also changing the type of vehicle after the warmup can be done manually here.
    """
    def __init__(self,
                 veh_id,
                 local_zone,
                 directory, 
                 policy_name, 
                 checkpoint_num,
                 num_cpus, # Make the num cpu here one more than the one in training config file for the single agent
                 warmup_steps,
                 efficiency = False, # For efficiency leader. Only at test time.
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 car_following_params=None,): 
        
        BaseController.__init__(
            self,
            veh_id,
            car_following_params)
        
        self.veh_id = veh_id

        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.WARMUP_STEPS = warmup_steps
        self.LOCAL_ZONE = local_zone
        self.MAX_SPEED = 10.0
        self.directory = directory
        self.policy_name = policy_name
        self.checkpoint_num = checkpoint_num
        self.num_cpus = num_cpus
        
        self.csc_model = self.load_csc_model()
        # get the leader agent ready 
        self.leader_agent = self.setup_trained_leader()

        # Efficiency specific
        self.efficiency = efficiency
        self.free_flow_speed = 0.0 

    def setup_trained_leader(self, ):
        """
        
        """
        import ray
        try:
            from ray.rllib.agents.agent import get_agent_class
        except ImportError:
            from ray.rllib.agents.registry import get_agent_class
        from flow.utils.registry import make_create_env
        from flow.utils.rllib import get_flow_params, get_rllib_config
        from ray.tune.registry import register_env

        result_dir_name = self.directory + self.policy_name
        ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True)

        result_dir = result_dir_name if result_dir_name[-1] != '/' else result_dir_name[:-1]
        checkpoint = result_dir + '/checkpoint_' + self.checkpoint_num
        checkpoint = checkpoint + '/checkpoint-' + self.checkpoint_num

        config = get_rllib_config(result_dir)
        config['num_workers'] = 0

        flow_params = get_flow_params(config)
        sim_params = flow_params['sim']
        setattr(sim_params, 'num_clients', 1)

        config_run = config['env_config']['run'] if 'run' in config['env_config'] else None
        agent_cls = get_agent_class(config_run)

        create_env, env_name = make_create_env(params=flow_params, version=0)
        register_env(env_name, create_env)

        agent = agent_cls(env=env_name, config=config)
        agent.restore(checkpoint)
        print(f"\n\nLeader agent restored\n{agent.get_policy()}")
        return agent
    
    def get_csc_output(self, current_obs):
        """
        Get the output of Traffic State Estimator Neural Network
        """
        current_obs = torch.from_numpy(current_obs).flatten()

        with torch.no_grad():
            outputs = self.csc_model(current_obs.unsqueeze(0))

        _, predicted_label = torch.max(outputs, 1)
        predicted_label = predicted_label.numpy()
        return predicted_label
    
    def load_csc_model(self, ):
        """
        Load the Traffic State Estimator Neural Network and its trained weights
        """
        class csc_Net(nn.Module):
            def __init__(self, input_size, num_classes):
                super(csc_Net, self).__init__() 
                self.fc1 = nn.Linear(input_size, 32)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(32, 16)
                self.relu = nn.ReLU()
                self.fc3 = nn.Linear(16, num_classes)
                
            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                out = self.relu(out)
                out = self.fc3(out)
                return out

        input_size = 10*2
        num_classes = 6
        url = "https://huggingface.co/matrix-multiply/Congestion_Stage_Classifier/resolve/main/ring_best_csc_model.pt"
        saved_best_net = csc_Net(input_size, num_classes)

        state_dict = torch.hub.load_state_dict_from_url(url)
        saved_best_net.load_state_dict(state_dict)
        saved_best_net.eval()

        return saved_best_net
    
    def get_idm_accel(self, env):
        """
        it will automatically call this for each vehicle
        At shock times, we have to return the shock acceleration
        """
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        #print("IDM")
        return self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)
    
    def get_trained_accel(self, env):
        """
        env: determines what functions below work how, for example, length() is valid for ring.
        Use the environment to get all the necessary things
        """
        
        # Get the observation for CSC input
        rl_pos = env.k.vehicle.get_x_by_id(self.veh_id)
        current_length = env.k.network.length() # WORKS FOR RING

        sorted_veh_ids = env.k.vehicle.get_veh_list_local_zone(self.veh_id, current_length, self.LOCAL_ZONE)
        sorted_veh_ids.remove(self.veh_id)
        sorted_veh_ids.insert(0, self.veh_id)

        observation_csc = np.full((10, 2), -1.0)
        for i in range(len(sorted_veh_ids)):
           rel_pos = (env.k.vehicle.get_x_by_id(sorted_veh_ids[i]) - rl_pos) % current_length
           norm_pos = rel_pos / self.LOCAL_ZONE

           vel = env.k.vehicle.get_speed(sorted_veh_ids[i])
           norm_vel = vel / self.MAX_SPEED
           observation_csc[i] = [norm_pos, norm_vel]

        observation_csc = np.array(observation_csc, dtype = np.float32)

        # Pass it and get CSC output
        csc_output = self.get_csc_output(observation_csc)
        csc_output_encoded = np.zeros(6) 
        csc_output_encoded[csc_output] = 1 

        # Leader of the current agent 
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        # normalizers
        max_speed = 15.
        max_length = current_length # THE SAME VALUE BUT DONE DIFFERENTLY HERE
        
        observation_regular = np.array([
                            env.k.vehicle.get_speed(self.veh_id) / max_speed,
                            (env.k.vehicle.get_speed(lead_id) -
                            env.k.vehicle.get_speed(self.veh_id)) / max_speed,
                            (env.k.vehicle.get_x_by_id(lead_id) -
                            env.k.vehicle.get_x_by_id(self.veh_id)) % current_length
                            / max_length
                            ])
        
        full_observation = np.append(observation_regular, csc_output_encoded)

        # Finally compute the action
        acceleration = self.leader_agent.compute_action(full_observation)
        #print(f"Acceleration: {acceleration}, Efficiency: {self.efficiency} \n")

        # Efficiency specific (Only present at test time)
        # Estimate the free flow speed
        if self.efficiency:
            
            if env.step_counter < self.WARMUP_STEPS + 200:
                # csc output is free flow 
                if csc_output[0] == 2:
                    estimate = 0.70*np.mean([env.k.vehicle.get_speed(veh_id) for veh_id in sorted_veh_ids]) # 0.70 for 20% and 40%, 0.78 for 60%,
                    if estimate > self.free_flow_speed:
                        self.free_flow_speed = estimate

            if env.step_counter > self.WARMUP_STEPS + 200 and env.k.vehicle.get_speed(self.veh_id) >= self.free_flow_speed:
                acceleration = 0.0

            #print("Estimated Free Flow Speed: ", self.free_flow_speed)
        return acceleration

    def get_accel(self, env):
        """
        manual implementation of the vehicle switch instead of TRACI
        """
        if env.step_counter < self.WARMUP_STEPS:
            return self.get_idm_accel(env)
        else: 
            return self.get_trained_accel(env)
        
