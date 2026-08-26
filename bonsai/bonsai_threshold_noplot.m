% bonsai_threshold_noplot.m — compute-only version (no figures, no saveas)
% Requires BCI_threshold and hSI in base workspace.
tic
global BCI_threshold BCI_params thr_hist thr_hist_session
hSI = evalin('base','hSI');

use_last_10_trials = true;

% ---- Find most recent Bonsai session on Z:\ ----
bonsai_root = 'Z:\';
a = dir(bonsai_root);
a = a([a.isdir] & ~ismember({a.name}, {'.', '..'}));
[~, idx] = max([a.datenum]);
bonsai_session = fullfile(bonsai_root, a(idx).name, 'behavior');
fprintf('Bonsai session: %s\n', bonsai_session);

% ---- Parse ResponsePeriod.json (go cue times) ----
raw = fileread(fullfile(bonsai_session, 'SoftwareEvents', 'ResponsePeriod.json'));
lines = strtrim(splitlines(raw));
lines = lines(cellfun(@(l) ~isempty(l) && l(end) == '}', lines));
go_times = cellfun(@(ln) jsondecode(ln).timestamp, lines);
fprintf('Found %d go cues\n', length(go_times));
if isempty(go_times), toc; return; end

% ---- Read SpoutPosition.csv ----
sp_path = fullfile(bonsai_session, 'OperationControl', 'SpoutPosition.csv');
fid = fopen(sp_path, 'r');
header = strsplit(fgetl(fid), ',');
header = strtrim(header);
raw = textscan(fid, repmat('%f', 1, length(header)), 'Delimiter', ',');
fclose(fid);
t_sp = raw{strcmp(header, 'Seconds')};
pos  = raw{strcmp(header, 'Value')};
P_max = max(pos);
fprintf('Spout samples: %d, P_max = %.1f mm\n', length(pos), P_max);

% ---- Compute time-to-reward per trial ----
hit_times = t_sp(pos >= P_max - 0.1);
t_end_all = [go_times(2:end); go_times(end) + 10];
tct = nan(1, length(go_times));
for i = 1:length(go_times)
    candidates = hit_times(hit_times >= go_times(i) & hit_times <= t_end_all(i));
    if ~isempty(candidates)
        tct(i) = candidates(1) - go_times(i);
    end
end
fprintf('Trials with hits: %d / %d\n', sum(~isnan(tct)), length(go_times));

% ---- Multiplier sweep ----
tct_first = tct(1:min(10, length(tct)));
if use_last_10_trials && length(tct) > 10
    tct = tct(end-9:end);
end

multipliers = linspace(1, 3, 500);

% Find first saved threshold file
num = 1;
old_thr = [];
while num < 100
    str = fullfile(hSI.hScan2D.logFilePath, ...
        [hSI.hScan2D.logFileStem, '_threshold_', num2str(num), '.mat']);
    if exist(str, 'file')
        tmp = load(str, 'BCI_threshold');
        old_thr = tmp.BCI_threshold;
        break;
    end
    num = num + 1;
end
if isempty(old_thr)
    error('Could not find any _threshold_*.mat file in %s', hSI.hScan2D.logFilePath);
end

multipliers_old = multipliers * diff(BCI_threshold) / diff(old_thr);

hit_rate = zeros(1, length(multipliers));
mean_time_to_hit = zeros(1, length(multipliers));
for i = 1:length(multipliers)
    hit_rate(i) = mean(tct * multipliers(i) < 10);
    vals = tct * multipliers(i); vals = vals(~isnan(vals));
    mean_time_to_hit(i) = mean(vals);
end

% ---- Determine recommended threshold ----
x_thr = BCI_threshold(1) + diff(BCI_threshold)*multipliers;
target_mth = mean_time_to_hit(1) + 1;
idx_2s = find(mean_time_to_hit >= target_mth, 1);

if ~isempty(idx_2s) && hit_rate(idx_2s) < 1
    idx_100 = find(hit_rate >= 1, 1, 'last');
    if ~isempty(idx_100)
        idx_rec = idx_100;
    else
        idx_rec = [];
    end
else
    idx_rec = idx_2s;
end

% ---- Threshold history (global, O(1) per call) ----
T = hSI.hScan2D.logFileCounter;
if ~isequal(thr_hist_session, hSI.hScan2D.logFileStem)
    thr_hist = [];
    thr_hist_session = hSI.hScan2D.logFileStem;
end
if isempty(thr_hist)
    thr_hist = BCI_threshold(2) * ones(1, T);
end
thr_hist(T) = BCI_threshold(2);
changed = find(diff(thr_hist(1:T)) ~= 0, 1, 'last');
if isempty(changed)
    last_change = T;
else
    last_change = T - changed;
end

% ---- Print recommendation ----
clc
hit = mean(~isnan(tct));
avgrt = nanmean(tct);
if last_change >= 10
    if hit == 1 && avgrt < 7
        disp(['Hit rate is 100% and average reward time is ', num2str(avgrt),'s'])
        disp(' ')
        if ~isempty(idx_rec)
            BCI_threshold(2) = x_thr(idx_rec);
            BCI_params.thr_low(1)  = BCI_threshold(1);
            BCI_params.thr_high(1) = BCI_threshold(2);
            if isfield(BCI_params, 'gui_handle') && isvalid(BCI_params.gui_handle)
                h = guidata(BCI_params.gui_handle);
                h.lower_1_edit.String = num2str(BCI_threshold(1));
                h.upper_1_edit.String = num2str(BCI_threshold(2));
            end
            if idx_rec == idx_2s
                disp(['Change threshold to ', num2str(x_thr(idx_rec)), char(10), ...
                    'to increase reward time by 1s'])
                disp(['Expected hit rate will be ', num2str(hit_rate(idx_rec)*100), '%'])
            else
                disp('Cannot increase reward time by 1s without dropping hit rate below 100%.')
                disp(['Change threshold to ', num2str(x_thr(idx_rec)), char(10), ...
                    'to push as hard as possible while keeping 100% hit rate'])
                disp(['Expected reward time: ', num2str(mean_time_to_hit(idx_rec)), 's'])
            end
        else
            disp('No valid threshold increase found — already at hit rate limit.')
        end
    end
    if hit == 1 && avgrt > 7
        ['Do not change threshold average reward time is ', num2str(avgrt),'s']
    end
    if hit < 1
        ['Do not change threshold hit rate is only ', num2str(hit*100),'%']
    end
else
    ['Do not change threshold last change was only ', num2str(last_change),' trials ago.']
end
toc
