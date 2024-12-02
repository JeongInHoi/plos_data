# common_2000_d0.py
#
# 기본 common.py를 두 가지 다른 stimuli_set들에 대하여 reward를 주며 Pavlovian 학습 진행 용도로 변경 
# (이경진 2023.02.02)
#
# 추가로 residual D0 값을 갖도록 변경
# (이경진 2023.02.25)
# 

import numpy as np
import matplotlib.pyplot as plt 

from pygenn import genn_model, genn_wrapper
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from six import iteritems

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------

izhikevich_dopamine_model = genn_model.create_custom_neuron_class(
    "izhikevich_dopamine",
 
    param_names=["a", "b", "c", "d", "tauD", "dStrength"],
    var_name_types=[("V", "scalar"), ("U", "scalar"), ("D", "scalar")],
    extra_global_params=[("rewardTimesteps", "uint32_t*")],
    sim_code=
        """
        $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT; //at two times for numerical stability
        $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT;
        $(U)+=$(a)*($(b)*$(V)-$(U))*DT;
        const unsigned int timestep = (unsigned int)($(t) / DT);
        const bool injectDopamine = (($(rewardTimesteps)[timestep / 32] & (1 << (timestep % 32))) != 0);
        if(injectDopamine) {
           const scalar dopamineDT = $(t) - $(prev_seT);
           const scalar dopamineDecay = exp(-dopamineDT / $(tauD));
           $(D) = ($(D) * dopamineDecay) + $(dStrength);
        }
        """,
    threshold_condition_code="$(V) >= 30.0",
    reset_code=
        """
        $(V)=$(c);
        $(U)+=$(d);
        """)


izhikevich_stdp_tag_update_code="""
    // Calculate how much tag has decayed since last update
    const scalar tagDT = $(t) - tc;
    const scalar tagDecay = exp(-tagDT / $(tauC));
    // Calculate how much dopamine has decayed since last update
    const scalar dopamineDT = $(t) - $(seT_pre);
    const scalar dopamineDecay = exp(-dopamineDT / $(tauD));
    // Calculate offset to integrate over correct area
    const scalar offset = (tc <= $(seT_pre)) ? exp(-($(seT_pre) - tc) / $(tauC)) : exp(-(tc - $(seT_pre)) / $(tauD));
    // Update weight and clamp
    // ****** 중요 아래 D_pre에 0.002를 추가했음 
    // 원래 주어진 식   $(g) += ($(c) * $(scale) * $(D_pre)) * ((tagDecay * dopamineDecay) - offset);
    $(g) += $(c) * $(scale) * ( $(D_pre)*((tagDecay * dopamineDecay) - offset))+ $(c)*$(tauC)*(1-tagDecay)*$(backgroundD);
    $(g) = fmax($(wMin), fmin($(wMax), $(g)));
    """
    
#***원래 파라미터  param_names=["tauPlus",  "tauMinus", "tauC", "tauD", "aPlus", "aMinus","wMin", "wMax"]
izhikevich_stdp_model = genn_model.create_custom_weight_update_class(
    "izhikevich_stdp",
    
    param_names=["tauPlus",  "tauMinus", "tauC", "tauD", "aPlus", "aMinus",
                 "wMin", "wMax", "backgroundD"],
    derived_params=[
        ("scale", genn_model.create_dpf_class(lambda pars, dt: 1.0 / -((1.0 / pars[2]) + (1.0 / pars[3])))())],
    var_name_types=[("g", "scalar"), ("c", "scalar")],

    sim_code=
        """
        $(addToInSyn, $(g));
        // Calculate time of last tag update
        const scalar tc = fmax($(prev_sT_pre), fmax($(prev_sT_post), $(prev_seT_pre)));
        """
        + izhikevich_stdp_tag_update_code +
        """
        // Decay tag and apply STDP
        scalar newTag = $(c) * tagDecay;
        const scalar dt = $(t) - $(sT_post);
        if (dt > 0) {
            scalar timing = exp(-dt / $(tauMinus));
            newTag -= ($(aMinus) * timing);
        }
        // Write back updated tag and update time
        $(c) = newTag;
        """,
    event_code=
        """
        // Calculate time of last tag update
        const scalar tc = fmax($(sT_pre), fmax($(prev_sT_post), $(prev_seT_pre)));
        """
        + izhikevich_stdp_tag_update_code +
        """
        // Decay tag
        $(c) *= tagDecay;
        """,
    learn_post_code=
        """
        // Calculate time of last tag update
        const scalar tc = fmax($(sT_pre), fmax($(prev_sT_post), $(seT_pre)));
        """
        + izhikevich_stdp_tag_update_code + 
        """
        // Decay tag and apply STDP
        scalar newTag = $(c) * tagDecay;
        const scalar dt = $(t) - $(sT_pre);
        if (dt > 0) {
            scalar timing = exp(-dt / $(tauPlus));
            newTag += ($(aPlus) * timing);
        }
        // Write back updated tag and update time
        $(c) = newTag;
        """,
    event_threshold_condition_code="injectDopamine",

    is_pre_spike_time_required=True, 
    is_post_spike_time_required=True,
    is_pre_spike_event_time_required=True,
    
    is_prev_pre_spike_time_required=True, 
    is_prev_post_spike_time_required=True,
    is_prev_pre_spike_event_time_required=True)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
# 정인회 background_D 추가, parser로 접근 가능
def get_params(size_scale_factor=1, build_model=True, 
               measure_timing=False, use_genn_recording=True):
    weight_scale_factor = 1.0 / size_scale_factor

    # Build params dictionary
    params = {
        "timestep_ms": 1.0,

        # Should we rebuild model
        "build_model": build_model,

        # Generate code for kernel timing
        "measure_timing": measure_timing,

        # Use GeNN's built in spike recording system
        "use_genn_recording": use_genn_recording,

        # Use zero copy for recording
        "use_zero_copy": False,
        
        # Simulation duration (*** 1st half에서만 자극을 가하므로 총 duration을 4배로 변경)
        "duration_ms": 60.0 * 60.0 * 1000.0 * 4,

        # How much of start and end of simulation to record
        # **NOTE** we want to see at least one rewarded stimuli in each recording window
        "record_time_ms": 12.0 * 1000.0,

        # How often should outgoing weights from each synapse be recorded
        "weight_record_interval_ms": 60.0 * 1000.0,

        # STDP params
        "tau_d": 200.0,

        # Scaled number of cells
       
        "num_excitatory": 1600 * int(size_scale_factor),
        "num_inhibitory": 400 * int(size_scale_factor),

        # Weights   ****
        "inh_weight": -8.0 * weight_scale_factor,  # *** originally -1.0   (NJP: -4.0) -8.0
        "init_exc_weight": 2.0 * weight_scale_factor, # *** originally 1.0 (NJP: ??) 1.5
        "max_exc_weight": 4.0 * weight_scale_factor, # originally 4.0 (NJP: 10.0) 4.0
        "dopamine_strength": 0.5 * weight_scale_factor, # *** originally 0.5  (none 0.000001)

        # Connection probability
        "probability_connection": 0.1, # *** originally 0.1

        # Input sets ****
        "num_stimuli_sets": 100,
        "stimuli_set_size": 100, # increased by a factor of 2 *** originally 50 for 1,000 node system
        "stimuli_current": 40.0, # *** originally 40.0

        # Regime
        "min_inter_stimuli_interval_ms": 100.0,
        "max_inter_stimuli_interval_ms": 300.0,

        # Reward
        "max_reward_delay_ms": 1000.0,
    	# background dopamine
    	"background_D": 0.0}
    # Loop through parameters
    dt = params["timestep_ms"]
    timestep_params = {}
    for n, v in iteritems(params):  # params dict에 있는 변수 기반 dt로 나뉘어진 새로운 변수 설정 
        # If parameter isn't timestep and it ends with millisecond suffix,
        # Add new version of parameter in timesteps to temporary dictionary
        if n != "timestep_ms" and n.endswith("_ms"):
            timestep_params[n[:-2] + "timestep"] = int(round(v / dt)) # "ms"를 "timestep"로 대체함 
    
    # Update parameters dictionary with new parameters and return
    params.update(timestep_params)
    return params

def plot_reward(axis, times):
    for t in times:
        axis.annotate("reward",
                      xy=(t, 0), xycoords="data",
                      xytext=(0, -15.0), textcoords="offset points",
                      arrowprops=dict(facecolor="black", headlength=6.0),
                      annotation_clip=True, ha="center", va="top")

def plot_stimuli(axis, times, num_cells):
    for t, i in times:
        colour = "green" if i == 0 else "black"
        axis.annotate("S%u" % i,
                      xy=(t, num_cells), xycoords="data",
                      xytext=(0, 15.0), textcoords="offset points",
                      arrowprops=dict(facecolor=colour, edgecolor=colour, headlength=6.0),
                      annotation_clip=True, ha="center", va="bottom", color=colour)

def convert_spikes(spike_list, timesteps):
    # Determine how many spikes were emitted in each timestep
    spikes_per_timestep = [len(s) for s in spike_list]
    assert len(timesteps) == len(spikes_per_timestep)

    # Repeat timesteps correct number of times to match number of spikes
    spike_times = np.repeat(timesteps, spikes_per_timestep)
    spike_ids = np.hstack(spike_list)
    
    return spike_times, spike_ids
    
def build_model(name, params, reward_timesteps):
    model = genn_model.GeNNModel("float", name)
    model.dT = params["timestep_ms"]
    model._model.set_merge_postsynaptic_models(True)
    model._model.set_default_narrow_sparse_ind_enabled(True)

    
    model.timing_enabled = params["measure_timing"]

    # Excitatory model parameters
    exc_params = {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0, 
                  "tauD": params["tau_d"], "dStrength": params["dopamine_strength"]}

    # Excitatory initial state
    exc_init = {"V": -65.0, "U": -13.0, "D": 0.0}

    # Inhibitory model parameters
    inh_params = {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0}

    # Inhibitory initial state
    inh_init = {"V": -65.0, "U": -13.0}

    # Static inhibitory synapse initial state
    inh_syn_init = {"g": params["inh_weight"]}

    # STDP parameters
    stdp_params = {"tauPlus": 20.0,  "tauMinus": 20.0, "tauC": 1000.0, 
                   "tauD": params["tau_d"], "aPlus": 0.12, "aMinus": 0.10, # *** originally, aPlus = 0.1,  aMinus = 0.15 
                   "backgroundD": params["background_D"],  
                   "wMin": 0.0, "wMax": params["max_exc_weight"]}

    # STDP initial state
    stdp_init = {"g": params["init_exc_weight"], "c": 0.0}
    
    # Fixed probability connector parameters
    fixed_prob_params = {"prob": params["probability_connection"]}
    
    # Create excitatory and inhibitory neuron populations
    e_pop = model.add_neuron_population("E", params["num_excitatory"], izhikevich_dopamine_model, 
                                        exc_params, exc_init)
    i_pop = model.add_neuron_population("I", params["num_inhibitory"], "Izhikevich", 
                                        inh_params, inh_init)
    
    # Turn on zero-copy for spikes if required
    if params["use_zero_copy"]:
        e_pop.pop.set_spike_location(genn_wrapper.VarLocation_HOST_DEVICE_ZERO_COPY)
        i_pop.pop.set_spike_location(genn_wrapper.VarLocation_HOST_DEVICE_ZERO_COPY)
    
    # Set dopamine timestep bitmask
    e_pop.set_extra_global_param("rewardTimesteps", reward_timesteps)

    # Enable spike recording
    e_pop.spike_recording_enabled = params["use_genn_recording"]
    i_pop.spike_recording_enabled = params["use_genn_recording"]

    # *** define delays
    e_delay = 0  # *** NJP 사용 평균 값: 3
    i_delay = 0  # *** NJP 사용 값: 1 
    
    # Add synapse population (*** originally genn wrapper.NO_DELAY, not replace by e_delay)
    e_e_pop = model.add_synapse_population("EE", "SPARSE_INDIVIDUALG", e_delay,
                                           "E", "E",
                                           izhikevich_stdp_model, stdp_params, stdp_init, {}, {},
                                           "DeltaCurr", {}, {},
                                           genn_model.init_connectivity("FixedProbability", fixed_prob_params))

    e_i_pop = model.add_synapse_population("EI", "SPARSE_INDIVIDUALG", e_delay,
                                 "E", "I",
                                 izhikevich_stdp_model, stdp_params, stdp_init, {}, {},
                                 "DeltaCurr", {}, {},
                                 genn_model.init_connectivity("FixedProbability", fixed_prob_params))

    # 이경진 named these two populations that had no variable assigned
    i_i_pop = model.add_synapse_population("II", "SPARSE_GLOBALG", i_delay,
                                 "I", "I",
                                 "StaticPulse", {}, inh_syn_init, {}, {},
                                 "DeltaCurr", {}, {},
                                 genn_model.init_connectivity("FixedProbability", fixed_prob_params))

    i_e_pop = model.add_synapse_population("IE", "SPARSE_GLOBALG", i_delay,
                                 "I", "E",
                                 "StaticPulse", {}, inh_syn_init, {}, {},
                                 "DeltaCurr", {}, {},
                                 genn_model.init_connectivity("FixedProbability", fixed_prob_params))
    
    # Return model and populations ** 여기에도 i_i_pop와 i_e_pop 추가 
    return model, e_pop, i_pop, e_e_pop, e_i_pop, i_i_pop, i_e_pop

def plot(start_exc_spikes, start_inh_spikes, transit_exc_spikes, transit_inh_spikes, middle_1_exc_spikes, middle_1_inh_spikes, middle_2_exc_spikes, middle_2_inh_spikes, middle_3_exc_spikes, middle_3_inh_spikes, end_exc_spikes, end_inh_spikes,
         start_stimulus_times, start_reward_times, 
         end_stimulus_times, end_reward_times, 
         display_time_ms, params):
    # Find the earliest rewarded stimuli in "start" and "end" recording data
    # *** 이경진 부연설명: s[1]는 dopamine reward를 받는 stimuli_set 인덱스, s[0]는 dopamine reward와 연관된 전기자극 전달 시각
    # 즉, 초입부와 마지막에 인덱스 0인 subgroup에 도파민해당 전기 자극이 들어간 시각들 확인 

    # 그러나 ...
    # 초입부 first_rewarded_stimuli_time_start = next(s[0] for s in start_stimulus_times if s[1] == 0)
    # 수정하여 말단부 두 다른 stimulus_set 용도로 바꿈 (예: 14와 13 set에 전달된 dopamine 전기 자극 (말단부 내) ) 
    first_rewarded_stimuli_time_start = next(s[0] for s in start_stimulus_times if s[1] == 15)
    first_rewarded_stimuli_time_end = next(s[0] for s in end_stimulus_times if s[1] == 15)

    # 위에서 "start"와 "end" 용어는 원래 초반부와 말단부를 의미
    
    # Find the corresponding reward (말단부 첫 도파민 자극 시점 각 set별로 확인) 
    # 즉, dopamine release용 첫 전기 자극 후, 나타난 첫 도파민 자극 시점 확인 
    corresponding_reward_time_start = next(r for r in start_reward_times if r > first_rewarded_stimuli_time_start)
    corresponding_reward_time_end = next(r for r in end_reward_times if r > first_rewarded_stimuli_time_end)

    # *** 이경진 부연설명: 전기자극 ~ 도파민 reward 시간 구역을 중심으로 2,000 ms range 좌우 시간 구역 설정  
    padding_start = (display_time_ms - (corresponding_reward_time_start - first_rewarded_stimuli_time_start)) / 2
    padding_end = (display_time_ms - (corresponding_reward_time_end - first_rewarded_stimuli_time_end)) / 2

    # Create plot (** graph 사이즈를 확장 정의를 추가함)
    figure, axes = plt.subplots(6, figsize=(12,18))

    # set the spacing between subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # set the spacing between subplots
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    num_cells = params["num_excitatory"] + params["num_inhibitory"]

    # plot the very initial time stage for Pavlovian learning    
    axes[0].scatter(start_exc_spikes[0], start_exc_spikes[1], s=2, edgecolors="none", color="red")
    axes[0].scatter(start_inh_spikes[0], start_inh_spikes[1] + params["num_excitatory"], s=2, edgecolors="none", color="blue")
    # **** 이경진 Let's save spiking data for the Pavlovian training time zome 
    np.savetxt("start_spiking_data.npy", start_exc_spikes)    
    # Plot reward times and rewarded stimuli that occur in first second
    plot_reward(axes[0], start_reward_times);
    plot_stimuli(axes[0], start_stimulus_times, num_cells)

    # plot the transit state (duration/48 time point) 
    axes[1].scatter(transit_exc_spikes[0], transit_exc_spikes[1], s=2, edgecolors="none", color="red")
    axes[1].scatter(transit_inh_spikes[0], transit_inh_spikes[1] + params["num_excitatory"], s=2, edgecolors="none", color="blue")
    # 
    np.savetxt("transit_spiking_data.npy", transit_exc_spikes)    
    # Plot reward times and rewarded stimuli 
    plot_reward(axes[1], end_reward_times);
    plot_stimuli(axes[1], end_stimulus_times, num_cells)

    # Plot spikes 
    axes[2].scatter(middle_1_exc_spikes[0], middle_1_exc_spikes[1], s=2, edgecolors="none", color="red")
    axes[2].scatter(middle_1_inh_spikes[0], middle_1_inh_spikes[1] + params["num_excitatory"], s=2, edgecolors="none", color="blue")
    # 
    np.savetxt("middle_1_spiking_data.npy", middle_1_exc_spikes)    
    # Plot reward times and rewarded stimuli 
    plot_reward(axes[2], end_reward_times);
    plot_stimuli(axes[2], end_stimulus_times, num_cells)
    # 

    # Plot spikes 
    axes[3].scatter(middle_2_exc_spikes[0], middle_2_exc_spikes[1], s=2, edgecolors="none", color="red")
    axes[3].scatter(middle_2_inh_spikes[0], middle_2_inh_spikes[1] + params["num_excitatory"], s=2, edgecolors="none", color="blue")
    # 
    np.savetxt("middle_2_spiking_data.npy", middle_2_exc_spikes)    

    # Plot spikes 
    axes[4].scatter(middle_3_exc_spikes[0], middle_3_exc_spikes[1], s=2, edgecolors="none", color="red")
    axes[4].scatter(middle_3_inh_spikes[0], middle_3_inh_spikes[1] + params["num_excitatory"], s=2, edgecolors="none", color="blue")
    # 
    np.savetxt("middle_3_spiking_data.npy", middle_3_exc_spikes)    

    # Plot spikes 
    axes[5].scatter(end_exc_spikes[0], end_exc_spikes[1], s=2, edgecolors="none", color="red")
    axes[5].scatter(end_inh_spikes[0], end_inh_spikes[1] + params["num_excitatory"], s=2, edgecolors="none", color="blue")
    # 
    np.savetxt("end_spiking_data.npy", end_exc_spikes)    

    # Configure axes
    # axes[0]과 axes[1]은 기본적으로 같은 데이터를 활용한 scatter plot이지만 보여지는 window range가 Pavlovian elec. 자극과
    # 도파민 자극 시점을 중심으로 해서 좌/우 대칭적으로 그려는 차이가 있다. 
    axes[0].set_title("Start", x=0.2, y=0.6)
    axes[1].set_title("Transit", x=0.2, y=0.6)
    axes[2].set_title("End", x=0.2, y=0.6)
    axes[3].set_title("No stimulation, time zone 1", x=0.2, y=0.6)
    axes[4].set_title("No stimulation, time zone 2", x=0.2, y=0.6)
    axes[5].set_title("No stimulation, time zone 3", x=0.2, y=0.6)    

    axes[0].set_xlim((first_rewarded_stimuli_time_start - padding_start, corresponding_reward_time_start + padding_start))
# *** axes[1] 시간 구역 설정 재 확인 필요  
    axes[1].set_xlim(( params["duration_ms"]*1/48 - 1 * display_time_ms, params["duration_ms"]*1/48 ))
    axes[2].set_xlim((first_rewarded_stimuli_time_end - padding_end, corresponding_reward_time_end + padding_end))
    axes[3].set_xlim(( params["duration_ms"]*2/4 - 1 * display_time_ms, params["duration_ms"]*2/4 ))
    axes[4].set_xlim(( params["duration_ms"]*3/4 - 1 * display_time_ms, params["duration_ms"]*3/4 ))    
    axes[5].set_xlim(( params["duration_ms"] - 1 * display_time_ms, params["duration_ms"] ))    

    print(first_rewarded_stimuli_time_start - padding_start)
    print(corresponding_reward_time_start + padding_start)
        
    axes[0].set_ylim((0, num_cells))
    axes[1].set_ylim((0, num_cells))
    axes[2].set_ylim((0, num_cells))    
    axes[3].set_ylim((0, num_cells))
    axes[4].set_ylim((0, num_cells))
    axes[5].set_ylim((0, num_cells))    

    axes[0].set_ylabel("Neuron number")
    axes[1].set_ylabel("Neuron number")
    axes[2].set_ylabel("Neuron number")
    axes[3].set_ylabel("Neuron number")
    axes[4].set_ylabel("Neuron number")
    axes[5].set_ylabel("Neuron number")    

    axes[0].set_xlabel("Time [ms]")
    axes[1].set_xlabel("Time [ms]")
    axes[2].set_xlabel("Time [ms]")    
    axes[3].set_xlabel("Time [ms]")
    axes[4].set_xlabel("Time [ms]")
    axes[5].set_xlabel("Time [ms]")        


