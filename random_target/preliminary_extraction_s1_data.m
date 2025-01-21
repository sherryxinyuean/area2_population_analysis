%% User inputs
load_data=1;

dt = .01; % in seconds. .01 = ` 10ms

vel_start_time=.7; %Time to have neural data prior to kinematics
vel_bin_size=.05; %Bin size for kinematics
neur_bin_size=.05; %Bin size for neural data

%% Load data

if load_data
    
    addpath(genpath('/Users/sherryan/area2_population_analysis/random-target/ClassyDataAnalysis-master'));
    load_folder='/Users/sherryan/area2_population_analysis/random_target/';
    load([load_folder 'Han_20160325_RW_hold_area2_001_cds'])
    
end

%% Get reaching directions
directions = nan(height(cds.trials),1);
tgtVar = cds.trials.tgtCtr;
for i = 2:height(cds.trials)
    x1 = tgtVar(i-1,1);
    y1 = tgtVar(i-1,2);
    x2 = tgtVar(i,1); 
    y2 = tgtVar(i,2);
    directions(i) = atan2d(y2-y1,x2-x1);
end
tgtVar = [tgtVar directions];

%% Times to use

col_start_time = 2;
col_end_time = 3;
col_go_cue_time = 5;

wdw_start = table2array(cds.trials([cds.trials.result]=='R',col_start_time));
% get rewarded trials
wdw_end = table2array(cds.trials([cds.trials.result]=='R',col_end_time));
wdw_go_cue = table2array(cds.trials([cds.trials.result]=='R',col_go_cue_time));

num_rewarded_trials = length(wdw_start);

trial_length = wdw_end - wdw_start;
good_trials_idx = find(trial_length >= 0.5 & trial_length <= 2.5);
num_good_trials = length(good_trials_idx);

edges=cell(num_good_trials,1); 
go_cue_bin = nan(num_good_trials,1);
for i = 1:num_good_trials
    trial_idx = good_trials_idx(i);
    edges{i} = wdw_start(trial_idx)-.5:dt:wdw_end(trial_idx)+.5;
    go_cue_bin(i)=find(wdw_start(trial_idx):dt:wdw_end(trial_idx)>=wdw_go_cue(trial_idx),1,'first');
end

tgtVar = tgtVar(good_trials_idx,:);
trial_length = trial_length(good_trials_idx,:);

%% Get neural data

% good_units=find([cds.units.ID]~=0 & [cds.units.ID]~=255);
good_units=find([cds.units.ID]~=255);
num_units=length(good_units);
good_neural_data = cell(num_good_trials,1);
for i = 1:num_good_trials
    edge = edges{i};
    temp = nan(num_units,length(edge)-1);
    for nrn_num = 1:num_units
        nrn_idx = good_units(nrn_num);
        temp(nrn_num,:) = histcounts(table2array(cds.units(nrn_idx).spikes(:,1)),edge);
    end
    good_neural_data{i} = temp.';
end

good_units=find([cds.units.ID]~=999);
num_units = width(cds.units);
all_neural_data = cell(num_good_trials,1);
for i = 1:num_good_trials
    edge = edges{i};
    temp = nan(num_units,length(edge)-1);
    for nrn_num = 1:num_units
        nrn_idx = good_units(nrn_num);
        temp(nrn_num,:) = histcounts(table2array(cds.units(nrn_idx).spikes(:,1)),edge);
    end
    all_neural_data{i} = temp.';
end

% length(neural_data_temp) = [num_good_trials]; 
% ith element has shape = [num_neurons * timepoint]

%% Get kinematics
wdw_start_good = wdw_start(good_trials_idx);
wdw_end_good = wdw_end(good_trials_idx);

position = cell(num_good_trials,1);
velocity = cell(num_good_trials,1);
acceleration = cell(num_good_trials,1);

diffs = nan(num_good_trials);
for i = 1:num_good_trials
    wdw_st = wdw_start_good(i);
    wdw_ed = wdw_end_good(i);
    [kin_start_idx]=find(table2array(cds.kin(:,1))>=wdw_st,1); %First kinematics time point in the window
    [kin_end_idx]=find(table2array(cds.kin(:,1))>=wdw_ed,1); %Last kinematics time point in the window
    diff = (kin_end_idx-kin_start_idx+1) - (length(wdw_st:dt:wdw_ed)-1);
    diffs(i) = diff;
    kin_end_idx=kin_end_idx-diff; % trim kinematics by difference with size of neural data
    position{i}= cat(2,table2array(cds.kin(kin_start_idx:kin_end_idx,4)),table2array(cds.kin(kin_start_idx:kin_end_idx,5)));
    velocity{i}= cat(2,table2array(cds.kin(kin_start_idx:kin_end_idx,6)),table2array(cds.kin(kin_start_idx:kin_end_idx,7)));
    acceleration{i}=cat(2,table2array(cds.kin(kin_start_idx:kin_end_idx,8)),table2array(cds.kin(kin_start_idx:kin_end_idx,9)));
    %Make kinematics same size as neural data
end

%% Save data
save_folder='/Users/sherryan/area2_population_analysis/random_target/';
%
save([save_folder 's1_all_units_trialized'],'good_neural_data','all_neural_data','position','velocity','acceleration','tgtVar','trial_length')


%% Put things in bins

num_vel_smooth_bins=vel_bin_size/dt;
vel_start_bin=vel_start_time/dt; %So it's centered

num_vel_bins=floor((length(x_vel)-vel_start_bin)/num_vel_smooth_bins);

num_neur_smooth_bins=neur_bin_size/dt;
num_neur_features=round(vel_start_time/neur_bin_size); %6 = 300 ms lag / 50 ms features


x_vel_binned=zeros(num_vel_bins,1);
y_vel_binned=zeros(num_vel_bins,1);
x_pos_binned=zeros(num_vel_bins,1);
y_pos_binned=zeros(num_vel_bins,1);
neur_data_binned=zeros(num_vel_bins,num_neur_features,num_units);
for i=1:num_vel_bins
    vel_start_idx=vel_start_bin+1+num_vel_smooth_bins*(i-1);
    vel_end_idx=vel_start_bin+num_vel_smooth_bins*(i);
    x_vel_binned(i)=mean(x_vel(vel_start_idx:vel_end_idx));
    y_vel_binned(i)=mean(y_vel(vel_start_idx:vel_end_idx));
    
    x_pos_binned(i)=mean(x_pos(vel_start_idx:vel_end_idx));
    y_pos_binned(i)=mean(y_pos(vel_start_idx:vel_end_idx));
    
    neur_start_idx_all=vel_start_idx-vel_start_bin;
    for j=1:num_neur_features
        neur_start_idx=neur_start_idx_all+num_neur_smooth_bins*(j-1);
        neur_end_idx=neur_start_idx_all+num_neur_smooth_bins*(j)-1;
        neur_data_binned(i,j,:)=mean(neural_data_temp(:,neur_start_idx:neur_end_idx),2);
        
    end
end


Y_vel=[x_vel_binned y_vel_binned];
Y_pos=[x_pos_binned y_pos_binned];
X=neur_data_binned;

%% Save

save_folder='/Users/jig289/Dropbox/MATLAB/Projects/In_Progress/BMI/Processed_Data';
%
% save([save_folder '/s1_data_' num2str(1000*vel_start_time) 'ms_' num2str(num_neur_features) '_neur_features_rnn_all.mat' ],'X','Y_vel','Y_pos')

