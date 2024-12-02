#  이 버젼은 common_PB_consolidation0308.py를 사용하는데 기존 common_PB_consolidation0303.py와 큰 차이는
# D0 값 (예: 0.002)를 함수 izhikevich_stdp_tag_upate_code에 있는 $(D_pre)에 더했다는 점이다.
# 앞서 ~0303.py 버젼에서는 izhikevich_dopamine_model에 $(D) = $(D) + 0.002의 형태로 더해졌었다. 
import os
import numpy as np
import matplotlib.pyplot as plt 
import csv
from argparse import ArgumentParser
from pygenn import genn_model, genn_wrapper
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from time import perf_counter

from common_PB_consolidation_basal_dopamine import (izhikevich_dopamine_model, izhikevich_stdp_model, 
                    build_model, get_params, plot, convert_spikes)
# 정인회 parser 추가, record_directory 추가
parser = ArgumentParser(add_help=True)
parser.add_argument("--background-D", type = float, default = 0.0, help = "background dopamine in izhikevich stdp model")
parser.add_argument("--dopamine-strength", type = float, default = 0.5, help = "dopamine injected as reward signal in izhikevich stdp model")
args = parser.parse_args()
base_record_directory = 'D'+str(args.background_D)
record_directory = base_record_directory
counter = 1
while os.path.exists(os.path.join('./simulation_result',record_directory)):
	record_directory = f"{base_record_directory}({counter})"
	counter +=1

record_directory = os.path.join('./simulation_result', record_directory)
os.mkdir(record_directory)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def get_start_end_stim(stim_counts):
    end_stimuli = np.cumsum(stim_counts)
    start_stimuli = np.empty_like(end_stimuli)
    start_stimuli[0] = 0
    start_stimuli[1:] = end_stimuli[0:-1]
    
    return start_stimuli, end_stimuli

# ----------------------------------------------------------------------------
# Write spike data to a file
# ----------------------------------------------------------------------------
def write_spike_file(filename, data):
    np.savetxt(os.path.join(record_directory,filename), np.column_stack(data), fmt=["%f","%d"], 
               delimiter=",", header="Time [ms], Neuron ID")

# ----------------------------------------------------------------------------
# Write stimuli to a file
def write_stimuli_file(filename, data_stimuli):
#    new_data_stimuli = [list(row) for row in zip(*data_stimuli)]
    np.savetxt(os.path.join(record_directory,filename), np.array(data_stimuli), delimiter=",", fmt=["%f","%d"])
#    np.savetxt(filename, np.array(data_stimuli), fmt=["%f","%d"])
#
# -----------------------------------------------------------------------------
# Write reward to a file
def write_rewards_file(filename, data_rewards):
#     np.savetxt(filename, np.column_stack(data_rewards), fmt=["%f"])
    np.savetxt(os.path.join(record_directory,filename), np.array(data_rewards), fmt=["%f"])
# ----------------------------------------------------------------------------

# Custom models
# ----------------------------------------------------------------------------
stim_noise_model = genn_model.create_custom_current_source_class(
    "stim_noise",

    param_names=["n", "stimMagnitude"],
    var_name_types=[("startStim", "unsigned int"), ("endStim", "unsigned int", VarAccess_READ_ONLY)],
    extra_global_params=[("stimTimes", "scalar*")],
    injection_code=
        """
        scalar current = ($(gennrand_uniform) * $(n) * 2.0) - $(n);
        if($(startStim) != $(endStim) && $(t) >= $(stimTimes)[$(startStim)]) {
           current += $(stimMagnitude);
           $(startStim)++;
        }
        $(injectCurrent, current);
        """)

# ----------------------------------------------------------------------------
# Stimuli generation
# ----------------------------------------------------------------------------
# Get standard model parameters
# 정인회 params[
params = get_params(build_model=True, measure_timing=False, use_genn_recording=True)

params["background_D"] = args.background_D
params["dopamine_strength"] = args.dopamine_strength

# Generate stimuli sets of neuron IDs
num_cells = params["num_excitatory"] + params["num_inhibitory"]
stim_gen_start_time =  perf_counter()
input_sets = [np.random.choice(num_cells, params["stimuli_set_size"], replace=False)
              for _ in range(params["num_stimuli_sets"])]

# Lists of stimulus and reward times for use when plotting
stimulus_times = []

reward_times = []


# Create list for each neuron
neuron_stimuli_times = [[] for _ in range(num_cells)]
total_num_exc_stimuli = 0
total_num_inh_stimuli = 0

# Create zeroes numpy array to hold reward timestep bitmask
# *** 32로 나눈 몫 -- 즉, 전체 계산 시간을 32 단위로 나눠서 reward 회수를 정하고 reward 시점을 설정코자 한 것 같음  -5/10 정인회, 전체 durtion_timestep을 32 단위로 나눈것은 맞고, reward  회수를 정한 것은 아니다. 32 씩 나뉘어진 각 구간에 reward_timestep 은 어떤 정수인데 (ex 87) 32 로 나뉘면 몫이 2이고 나머지가 23이므로 2번째 구간(0,1,2....째 구간으로 생각할때) 에 위치한다. 여기서 2번째 구간이란  reward_timesteps 라는 array 의 2번째 위치를 의미한다. 이 2번째 위치는 2^32 까지의 정수를 담을 수 있다.(dtype = np.uint32 이다.) 앞서말한 나머지 23이 2^23으로 바뀌어 2번째 위치에 저장된다. 동일한 방법으로 64 는 2번째 구간에 2^0으로 저장된다. 65는 2번째 구간에 2^1로 저장된다. 즉 숫자하나로 각 구간의 reward_timestep여러개를 엔코딩하기위해 이런 방법을 채택한것 같다. (64,65,87 을 2^23+2^1+2^0 인 한개의 숫자로 표현할 수 있다.)
reward_timesteps = np.zeros((params["duration_timestep"] + 31) // 32, dtype=np.uint32)

# Loop while stimuli are within simulation duration

next_stimuli_timestep = np.random.randint(params["min_inter_stimuli_interval_timestep"],
                                          params["max_inter_stimuli_interval_timestep"])
# *** 아래 조건문에서 1st quater에서만 전기 + 도파민 자극 가함으로 변경  
while next_stimuli_timestep < params["duration_timestep"]/4:
    # Pick a stimuli set to present at this timestep
    stimuli_set = np.random.randint(params["num_stimuli_sets"])

    # Loop through neurons in stimuli set and add time to list
    for n in input_sets[stimuli_set]:
        neuron_stimuli_times[n].append(next_stimuli_timestep * params["timestep_ms"])

    # Count the number of excitatory neurons in input set and add to total
    num_exc_in_input_set = np.sum(input_sets[stimuli_set] < params["num_excitatory"])
    total_num_exc_stimuli += num_exc_in_input_set
    total_num_inh_stimuli += (num_cells - num_exc_in_input_set)

    # If we should be recording at this point, add stimuli to list
    
    #정인회  전부 record 할것이므로, start_stimulus_times 와 end_stimulus_times를 구분하지 않고 stimulus_times로 바꾸었다. start_reward_times 와 end_reward_times도 reward_times로 바꿨다.
    stimulus_times.append((next_stimuli_timestep * params["timestep_ms"], stimuli_set))
    
    

    # If this is the rewarded stimuli

    # ##############  *** original condition 은 stimuli_set == 0 하나만 ... 
    # if stimuli_set == 0:

    if ((stimuli_set == 15) or (stimuli_set == 15)):

        # Determine time of next reward
        reward_timestep = next_stimuli_timestep + np.random.randint(params["max_reward_delay_timestep"])

        # If this is within simulation
        # *** 여기서도 1st quater duration으로 바꿈 
        if reward_timestep < params["duration_timestep"]/4:
            # Set bit in reward timesteps bitmask
            reward_timesteps[reward_timestep // 32] |= (1 << (reward_timestep % 32))

            # If we should be recording at this point, add reward to list
            
            reward_times.append(reward_timestep * params["timestep_ms"])
            

    # Advance to next stimuli
    next_stimuli_timestep += np.random.randint(params["min_inter_stimuli_interval_timestep"],
                                               params["max_inter_stimuli_interval_timestep"])

# Count stimuli each neuron should emit
neuron_stimuli_counts = [len(n) for n in neuron_stimuli_times]

stim_gen_end_time = perf_counter()
print("Stimulus generation time: %fms" % ((stim_gen_end_time - stim_gen_start_time) * 1000.0))

# ----------------------------------------------------------------------------
# Network creation
# ----------------------------------------------------------------------------
# Assert that duration is a multiple of record time
assert (params["duration_timestep"] % params["record_time_timestep"]) == 0

# ** 이경진    i_i_pop and i_e_pop  둘 추가함 
# Build base model
model, e_pop, i_pop, e_e_pop, e_i_pop, i_i_pop, i_e_pop = build_model("izhikevich_pavlovian_gpu_stim", 
                                                    params, reward_timesteps)
# Current source parameters  # 정인회 6.5 originally
curr_source_params = {"n": 5.5, "stimMagnitude": params["stimuli_current"]}

# Calculate start and end indices of stimuli to be injected by each current source
start_exc_stimuli, end_exc_stimuli = get_start_end_stim(neuron_stimuli_counts[:params["num_excitatory"]])
start_inh_stimuli, end_inh_stimuli = get_start_end_stim(neuron_stimuli_counts[params["num_excitatory"]:])

# Current source initial state
exc_curr_source_init = {"startStim": start_exc_stimuli, "endStim": end_exc_stimuli}
inh_curr_source_init = {"startStim": start_inh_stimuli, "endStim": end_inh_stimuli}

# Add background current sources
e_curr_pop = model.add_current_source("ECurr", stim_noise_model, "E", 
                                      curr_source_params, exc_curr_source_init)
i_curr_pop = model.add_current_source("ICurr", stim_noise_model, "I", 
                                      curr_source_params, inh_curr_source_init)

# Set stimuli times
e_curr_pop.set_extra_global_param("stimTimes", np.hstack(neuron_stimuli_times[:params["num_excitatory"]]))
i_curr_pop.set_extra_global_param("stimTimes", np.hstack(neuron_stimuli_times[params["num_excitatory"]:]))

if params["build_model"]:
    print("Building model")
    model.build()

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
# Load model, allocating enough memory for recording
print("Loading model")
model.load(num_recording_timesteps=params["duration_timestep"])

print("Simulating")
# Loop through timesteps
# start_exc_spikes ... end_inh_spikes를 따로 저장할 필요가 없다. 전부 통합해  exc_spikes와 inh_spikes로 바꾸자.
sim_start_time =  perf_counter()
exc_spikes = None if params["use_genn_recording"] else []
inh_spikes = None if params["use_genn_recording"] else []
# connectivity 출력 ############################################################
no_exc = params["num_excitatory"]

e_e_pop.pull_connectivity_from_device()
ind_e_e= np.vstack((e_e_pop.get_sparse_pre_inds(), e_e_pop.get_sparse_post_inds()))
np.save(record_directory+'/'+"ind_e_e_"+str(1)+".npy", ind_e_e)
e_i_pop.pull_connectivity_from_device()
ind_e_i = np.vstack((e_i_pop.get_sparse_pre_inds(), e_i_pop.get_sparse_post_inds()+no_exc))
np.save(record_directory+'/'+"ind_e_i_"+str(1)+".npy", ind_e_i)
i_i_pop.pull_connectivity_from_device()
ind_i_i = np.vstack((i_i_pop.get_sparse_pre_inds()+no_exc, i_i_pop.get_sparse_post_inds()+no_exc))
np.save(record_directory+'/'+"ind_i_i_"+str(1)+".npy", ind_i_i)
i_e_pop.pull_connectivity_from_device()
ind_i_e = np.vstack((i_e_pop.get_sparse_pre_inds()+no_exc, i_e_pop.get_sparse_post_inds()))
np.save(record_directory+'/'+"ind_i_e_"+str(1)+".npy", ind_i_e)

while model.t < params["duration_ms"]:
    # Simulation
    model.step_time()

    if params["use_genn_recording"]:
        # If we've just finished simulating the initial recording interval
        if (model.timestep % params["record_time_timestep"]) ==0 :
            # Download recording data
            record_number = str(model.timestep // params["record_time_timestep"])
            
	    #e_curr_pop.pull_var_from_device()
	    #i_curr_pop.pull_var_from_device()
	    



            # edge들에 해당되는 weight 값들 확인 출력 #####################################
            e_e_pop.pull_var_from_device("g")
            g_e_e = e_e_pop.get_var_values("g")
            np.save(record_directory+'/'+"g_e_e_" + record_number+".npy", g_e_e, allow_pickle=True)

            e_i_pop.pull_var_from_device("g")
            g_e_i = e_i_pop.get_var_values("g")
            np.save(record_directory+'/'+"g_e_i_" + record_number+".npy", g_e_i, allow_pickle=True)
            
            '''
            i_e_pop.pull_var_from_device("g")
            g_i_e = i_e_pop.get_var_values("g")
            np.save("g_i_e_" + record_number+".npy", g_i_e, allow_pickle=True)
            
            
            i_i_pop.pull_var_from_device("g")
            g_i_i = i_i_pop.get_var_values("g")
            np.save("g_i_i_" + record_number+".npy", g_i_i, allow_pickle=True)
            
            g_tot = np.concatenate((g_e_e, g_e_i, g_i_e, g_i_i))
            np.save("g_tot_"+ record_number+".npy", g_tot)
            '''
            # #########################################################################
	

	
    else:
        # 학습이 시작되는 시점 
        if model.timestep <= params["record_time_timestep"]:
            e_pop.pull_current_spikes_from_device()
            i_pop.pull_current_spikes_from_device()
            start_exc_spikes.append(np.copy(e_pop.current_spikes))
            start_inh_spikes.append(np.copy(i_pop.current_spikes))
        # 학습 duration의 반이 지난 시점 
        elif model.timestep > (params["duration_timestep"]*1/48 - params["record_time_timestep"]):
            e_pop.pull_current_spikes_from_device()
            i_pop.pull_current_spikes_from_device()
            transit_exc_spikes.append(np.copy(e_pop.current_spikes))
            transit__inh_spikes.append(np.copy(i_pop.current_spikes))
        # 학습이 완료된 시점 
        elif model.timestep > (params["duration_timestep"]*1/4 - params["record_time_timestep"]):
            e_pop.pull_current_spikes_from_device()
            i_pop.pull_current_spikes_from_device()
            middle_1_exc_spikes.append(np.copy(e_pop.current_spikes))
            middle_1_inh_spikes.append(np.copy(i_pop.current_spikes))
        # Free running 1구역 지나는 시점
        elif model.timestep > (params["duration_timestep"]*2/4 - params["record_time_timestep"]):
            e_pop.pull_current_spikes_from_device()
            i_pop.pull_current_spikes_from_device()
            middle_2_exc_spikes.append(np.copy(e_pop.current_spikes))
            middle_2_inh_spikes.append(np.copy(i_pop.current_spikes))
        # Free running 2구역 지나는 시점
        elif model.timestep > (params["duration_timestep"]*3/4 - params["record_time_timestep"]):
            e_pop.pull_current_spikes_from_device()
            i_pop.pull_current_spikes_from_device()
            middle_3_exc_spikes.append(np.copy(e_pop.current_spikes))
            middle_3_inh_spikes.append(np.copy(i_pop.current_spikes))
        # Free running 끝나는 시점
        elif model.timestep > (params["duration_timestep"] - params["record_time_timestep"]):
            e_pop.pull_current_spikes_from_device()
            i_pop.pull_current_spikes_from_device()
            end_exc_spikes.append(np.copy(e_pop.current_spikes))
            end_inh_spikes.append(np.copy(i_pop.current_spikes))

model.pull_recording_buffers_from_device()
exc_spikes = e_pop.spike_recording_data
inh_spikes = i_pop.spike_recording_data     
       
sim_end_time =  perf_counter()

# write spikes data to csv files 
# 참고로 path 설정은 이런 식으로 .... write_spike_file(os.path.join(output_directory, "recurrent_spikes_%u_%u_%u.csv" % (epoch, batch_idx, rank_offset + i)), s)

# 추가로 .csv 파일 형식으로 spike data 출력

write_spike_file("izhikevich_e_spikes.csv", exc_spikes)
write_spike_file("izhikevich_i_spikes.csv", inh_spikes)

# Pavelovian 자극이 들어가는 1st quater 
write_stimuli_file("izhikevich_stimuli_times.csv", stimulus_times)
write_rewards_file("izhikevich_rewards_times.csv", reward_times)

# 입력을 받는 첫 번째 세트 소속 노드들을 출력 ##################################
np.save(record_directory+'/'+"input_nodes_15.npy", input_sets[15])
#np.save("input_nodes_55.npy", input_sets[55])

print("Simulation time: %fms" % ((sim_end_time - sim_start_time) * 1000.0))

if not params["use_genn_recording"]:
    start_timesteps = np.arange(0.0, params["record_time_ms"], params["timestep_ms"])
    end_timesteps = np.arange(params["duration_ms"] - params["record_time_ms"], params["duration_ms"], params["timestep_ms"])

    start_exc_spikes = convert_spikes(start_exc_spikes, start_timesteps)
    start_inh_spikes = convert_spikes(start_inh_spikes, start_timesteps)
    end_exc_spikes = convert_spikes(end_exc_spikes, end_timesteps)
    end_inh_spikes = convert_spikes(end_inh_spikes, end_timesteps)

if params["measure_timing"]:
    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tPresynaptic update:%f" % (1000.0 * model.presynaptic_update_time))
    print("\tPostsynaptic update:%f" % (1000.0 * model.postsynaptic_update_time))

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
'''
plot(start_exc_spikes, start_inh_spikes, transit_exc_spikes, transit_inh_spikes, middle_1_exc_spikes, middle_1_inh_spikes, middle_2_exc_spikes, middle_2_inh_spikes, middle_3_exc_spikes, middle_3_inh_spikes, end_exc_spikes, end_inh_spikes,
     start_stimulus_times, start_reward_times, 
     end_stimulus_times, end_reward_times,
     4000.0, params)

# Show plot
plt.show()
'''

