% cn_activity_monitor.m — per-trial average CN activity during experiment
% Requires hSI, hSICtl, BCI_params in base workspace.
tic
global BCI_params
hSI    = evalin('base', 'hSI');
hSICtl = evalin('base', 'hSICtl');

base = hSI.hScan2D.logFileStem;
cn   = find([hSICtl.hGUIData.integrationRoiOutputChannelControlsV5.tbRoiSelection.Data{:,1}] == 1);
dt_si = 1 / hSI.hRoiManager.scanVolumeRate;

% ---- Pull from in-memory trace ----
frames_per_trial = BCI_params.(base).frames_per_trial;
n_trials         = length(frames_per_trial);
if n_trials == 0, disp('No completed trials yet.'); return; end

cn_trace = BCI_params.(base).cn_trace(1:sum(frames_per_trial));
fprintf('Trials: %d\n', n_trials);

% ---- Find most recent Bonsai session ----
bonsai_root = 'Z:\';
a = dir(bonsai_root);
a = a([a.isdir] & ~ismember({a.name}, {'.', '..'}));
[~, idx] = max([a.datenum]);
bonsai_session = fullfile(bonsai_root, a(idx).name, 'behavior');

% ---- Parse ResponsePeriod.json ----
raw_json = fileread(fullfile(bonsai_session, 'SoftwareEvents', 'ResponsePeriod.json'));
lines = strtrim(splitlines(raw_json));
lines = lines(cellfun(@(l) ~isempty(l) && l(end) == '}', lines));
go_times = cellfun(@(ln) jsondecode(ln).timestamp, lines);
if isempty(go_times), disp('No go cues found.'); return; end

% ---- Read SpoutPosition.csv ----
sp_path = fullfile(bonsai_session, 'OperationControl', 'SpoutPosition.csv');
fid = fopen(sp_path, 'r');
sp_header = strtrim(strsplit(fgetl(fid), ','));
sp_raw = textscan(fid, repmat('%f', 1, length(sp_header)), 'Delimiter', ',');
fclose(fid);
t_sp  = sp_raw{strcmp(sp_header, 'Seconds')};
pos   = sp_raw{strcmp(sp_header, 'Value')};
P_max = max(pos);

% ---- Compute tct ----
hit_times = t_sp(pos >= P_max - 0.1);
t_end_all = [go_times(2:end); go_times(end) + 10];
tct = nan(1, length(go_times));
for i = 1:length(go_times)
    candidates = hit_times(hit_times >= go_times(i) & hit_times <= t_end_all(i));
    if ~isempty(candidates)
        tct(i) = candidates(1) - go_times(i);
    end
end
fprintf('Hits: %d / %d\n', sum(~isnan(tct)), length(go_times));

% ---- Per-trial mean CN activity, cut at tct for hits ----
cn_mean   = nan(1, n_trials);
n_matched = min(n_trials, length(tct));
row = 1;
for i = 1:n_matched
    seg = cn_trace(row : row + frames_per_trial(i) - 1);
    if ~isnan(tct(i))
        cut = min(round(tct(i) / dt_si), length(seg));
        seg = seg(1:cut);
    end
    cn_mean(i) = nanmean(seg);
    row = row + frames_per_trial(i);
end

% ---- Threshold change markers ----
thr_hist    = BCI_params.(base).current_thresholds(2, 1:sum(frames_per_trial));
trial_ends  = cumsum(frames_per_trial);
thr_at_trial = thr_hist(trial_ends);
thr_changes  = find(diff(thr_at_trial) ~= 0);

% ---- Plot ----
figure(1339); clf;
plot(1:n_trials, cn_mean, 'ko-', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
hold on;
for i = 1:length(thr_changes)
    xline(thr_changes(i), 'r--');
end
xlabel('Trial #')
ylabel('Mean CN fluorescence')
title(base, 'Interpreter', 'none')
xlim([1 max(n_trials, 2)])
toc
