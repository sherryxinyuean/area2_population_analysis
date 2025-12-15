% REFORMAT_CO_DATASET
% This script loads the original bumpmove dataset, adds the ctrHoldBump label,
% and saves the reformatted version.

clear; clc;

%% Input/output filenames
file_path = "/Volumes/TOSHIBA EXT/sherry/area2/duncan/";
infile = 'Duncan_20191016_CObumpmove_10ms.mat';
outfile = 'Duncan_20191016_COactpas_reformat';

%% Load data
S = load(file_path+infile, 'trial_data');
trial_data = S.trial_data;

%% --- Classify trials using start/end times ---

nTrials   = numel(trial_data.trialID);

idx_start = trial_data.idx_startTime(:);      % nTrials x 1
idx_end   = trial_data.idx_endTime(:);        % nTrials x 1

idx_bump_all  = trial_data.idx_bumpTime(:);   % nBumps x 1 (global time indices)
idx_go_all    = trial_data.idx_goCueTime(:);  % nGoCues x 1 (global time indices)

bumpDir = trial_data.bumpDir(:);              % nTrials x 1 (NaN = no bump)

% Our label:
%   0   = active (no bump)
%   1   = passive / center-hold bump  (bump before go cue)
%   NaN = reach-bump (bump after go cue) OR ambiguous
ctrHoldBump = nan(1,nTrials);

for tr = 1:nTrials
    t0 = idx_start(tr);
    t1 = idx_end(tr);

    % all bump times that fall within this trial
    bumps_this_trial = idx_bump_all( ...
        idx_bump_all >= t0 & idx_bump_all <= t1);

    % all go-cue times that fall within this trial
    gocues_this_trial = idx_go_all( ...
        idx_go_all >= t0 & idx_go_all <= t1);

    if isempty(bumps_this_trial)
        % no bump at all -> active trial
        ctrHoldBump(tr) = 0;
        continue
    end

    % there is at least one bump in this trial
    if isempty(gocues_this_trial)
        % bump but no go-cue registered: ambiguous; keep as NaN
        % (you could decide to treat these as center-hold if you want)
        continue
    end

    % compare *first* bump time to *first* go-cue time in this trial
    if min(bumps_this_trial) < min(gocues_this_trial)
        % bump occurs before go cue -> center-hold / passive
        ctrHoldBump(tr) = 1;
    else
        % bump occurs after go cue -> reach-bump; leave as NaN
    end
end

% fix_bd = isnan(bumpDir) & ~(ctrHoldBump == 0);  % NaN bumpDir but found bump
% if any(fix_bd)
%     warning('Some trials disagree between bumpDir==NaN and ctrHoldBump==0 (%d mismatches).', ...
%         sum(fix_bd));
% end
% bumpDir(fix_bd) = 999;    % or any placeholder indicating “unknown bump”
% trial_data.bumpDir = bumpDir;

%% Add field and save
trial_data.ctrHoldBump = ctrHoldBump;

save(file_path+outfile, 'trial_data', '-v7.3');

disp('Done! Saved reformatted dataset as:');
disp(outfile);