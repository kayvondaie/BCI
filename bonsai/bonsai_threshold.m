% bonsai_threshold.m — standalone version of bpod_threshold callback
% Reads Bonsai session data from Z:\ and plots threshold sweep.
% Requires BCI_threshold and hSI in base workspace.

global BCI_threshold
hSI = evalin('base','hSI');

use_last_10_trials = true;

% ---- Find most recent Bonsai session on Z:\ ----
bonsai_root = 'Z:\';
a = dir(bonsai_root);
a = a(3:end);
a = a([a.isdir]);
folders = sort({a.name}');
bonsai_session = fullfile(bonsai_root, folders{end}, 'behavior');
fprintf('Bonsai session: %s\n', bonsai_session);

% ---- Parse ResponsePeriod.json (go cue times) ----
fid = fopen(fullfile(bonsai_session, 'SoftwareEvents', 'ResponsePeriod.json'), 'r');
go_times = [];
while ~feof(fid)
    ln = fgetl(fid);
    if isempty(strtrim(ln)); continue; end
    ev = jsondecode(ln);
    go_times(end+1) = ev.timestamp;
end
fclose(fid);
fprintf('Found %d go cues\n', length(go_times));

% ---- Parse SpoutPosition.csv to get reward times ----
sp = readtable(fullfile(bonsai_session, 'OperationControl', 'SpoutPosition.csv'));
sp.Properties.VariableNames = strtrim(sp.Properties.VariableNames);
t_sp = sp.Seconds;
pos = sp.Value;
P_max = max(pos);
fprintf('Spout samples: %d, P_max = %.1f mm\n', length(pos), P_max);

% ---- Compute time-to-reward per trial ----
tct = nan(1, length(go_times));
for i = 1:length(go_times)
    t_go = go_times(i);
    if i < length(go_times)
        t_end = go_times(i+1);
    else
        t_end = t_sp(end);
    end
    hit_idx = find(t_sp >= t_go & t_sp <= t_end & pos >= P_max - 0.1, 1);
    if ~isempty(hit_idx)
        tct(i) = t_sp(hit_idx) - t_go;
    end
end
fprintf('Trials with hits: %d / %d\n', sum(~isnan(tct)), length(tct));

% ---- Multiplier sweep ----
tctt = 1:length(tct);
tct_first = tct(1:min(10, length(tct)));
tctt_first = tctt(1:min(10, length(tct)));
if use_last_10_trials && length(tct) > 10
    tct = tct(end-9:end);
    tctt = tctt(end-9:end);
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
        fprintf('Loaded initial threshold from: %s\n', str);
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
hit_rate_first = zeros(1, length(multipliers));
mean_time_to_hit_first = zeros(1, length(multipliers));
for i = 1:length(multipliers)
    hit_rate(i) = mean(tct * multipliers(i) < 10);
    vals = tct * multipliers(i); vals = vals(~isnan(vals));
    mean_time_to_hit(i) = mean(vals);
    hit_rate_first(i) = mean(tct_first * multipliers_old(i) < 10);
    vals2 = tct_first * multipliers_old(i); vals2 = vals2(~isnan(vals2));
    mean_time_to_hit_first(i) = mean(vals2);
end

figure(1338); clf;
subplot(3,1,1)
plot(tctt, tct, 'ko'); hold on;
xlabel('trial#')
ylabel('time to reward (s)')
ylim([0, 10])

subplot(3,1,2)
plot(BCI_threshold(1) + diff(BCI_threshold)*multipliers, hit_rate, 'k'); hold on;
plot(old_thr(1) + diff(old_thr)*multipliers_old, hit_rate_first, 'color', [.5 .5 .5])
xlabel('New high threshold')
ylabel('hit rate')
ylim([0, 1])

subplot(3,1,3)
plot(BCI_threshold(1) + diff(BCI_threshold)*multipliers, mean_time_to_hit, 'k'); hold on;
plot(old_thr(1) + diff(old_thr)*multipliers_old, mean_time_to_hit_first, 'color', [.5 .5 .5])
xlabel('New high threshold')
ylabel('time to hit')

str = fullfile(hSI.hScan2D.logFilePath, ...
    [hSI.hScan2D.logFileStem, '_threshold_calc_', num2str(hSI.hScan2D.logFileCounter), '.png']);
saveas(figure(1338), str);
fprintf('Saved %s\n', str);
