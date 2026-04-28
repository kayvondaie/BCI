figure(9495)
global BCI_threshold BCI_params

base = hSI.hScan2D.logFileStem;
cn = find([hSICtl.hGUIData.integrationRoiOutputChannelControlsV5.tbRoiSelection.Data{:,1}]==1);
folder = [hSI.hScan2D.logFilePath,'\'];
folder_props = folder_props_fun(folder);
files = dir(folder);
csv_files = {files(cell2mat(cellfun(@(x)~isempty(regexp(x,'.csv')),...
    {files.name},'UniformOutput', false))==1).name};
csv_files = {csv_files{cell2mat(cellfun(@(x)~isempty(strfind(x,'IntegrationRois')),csv_files,'uni', false))}}';
csv_files = {csv_files{cell2mat(cellfun(@(x)~isempty(strfind(x,base)),csv_files,'uni', false))}}';

clear file_time bases;
for i = 1:length(csv_files)
    file = [folder,csv_files{i}];
    file_info = dir(file);
    ind = min(find(csv_files{i}=='_'));
    bases{i} = csv_files{i}(1:ind-1);
    file_time(i) = datenum(file_info.date);
end
newest = find(file_time == max(file_time));
current_base = bases{newest};
inds = cellfun(@(x) strcmp(x,current_base),bases,'uni',0);
csv_files = csv_files([inds{:}]);
ROI = [];
for i = 1:1;
    csv_file = [folder,csv_files{i}];
    csvdata = readtable(csv_file);
    roi = table2array(csvdata);
    if i == 1;
        ROI = roi;
    else
        ROI = [ROI;roi];
    end
end

valueHistory = roi(:,3:end);

%% ---- Parameters ----
V_max         = 3.3;    % V (max voltage output)
integral_hit  = 0.58;   % V·s — voltage integral needed to traverse full distance
                        % (measured from fit_k.py: consistent to 3% CV across trials)
response_T    = 10;     % s — response period duration
dt_si = 1/hSI.hRoiManager.scanVolumeRate;  % seconds per frame

%% ---- Threshold sweep ----
cn = find([hSICtl.hGUIData.integrationRoiOutputChannelControlsV5.tbRoiSelection.Data{:,1}]==1);
h = valueHistory(:,cn);

% Set initial FL/FU so that predicted hit rate ~ 50%
BCI_threshold(1) = prctile(h, 20);
BCI_threshold(2) = BCI_threshold(1) + (prctile(h,100) - BCI_threshold(1)) * 0.5;
BCI_threshold = BCI_threshold';

new_thr = BCI_threshold(1)*linspace(.5,2,20);   % FL grid
new_slp = BCI_threshold(2)*linspace(.25,2,20);   % FU grid
n_frames_trial = round(response_T / dt_si);      % frames in one response period

% For each (FL, FU), simulate 100 random trial starts and check whether
% cumulative ∫V_m dt reaches integral_hit within response_T
hit_rate = nan(length(new_thr), length(new_slp));
median_T = nan(length(new_thr), length(new_slp));
for ti = 1:length(new_thr)
    FL = new_thr(ti);
    for si = 1:length(new_slp)
        FU = new_slp(si);
        if FU <= FL; continue; end
        hits = 0;
        T_hits = nan(100,1);
        for i = 1:100
            strt = randsample(size(valueHistory,1) - n_frames_trial - 1, 1);
            seg = h(strt:strt + n_frames_trial - 1);
            % Fluorescence -> voltage (matches BCI_analog_display.m)
            Vm = max(seg - FL, 0) .* (V_max / (FU - FL));
            Vm = min(Vm, V_max);
            % Cumulative voltage integral (V·s)
            cum_integral = cumsum(Vm) * dt_si;
            % Find first frame where integral reaches threshold
            hit_frame = find(cum_integral >= integral_hit, 1);
            if ~isempty(hit_frame)
                hits = hits + 1;
                T_hits(i) = hit_frame * dt_si;
            end
        end
        hit_rate(ti,si) = hits / 100;
        median_T(ti,si) = nanmedian(T_hits);
    end
end

%% ---- Plot: hit rate ----
figure(9495); clf

imagesc(hit_rate, [0 1]); hold on;
colorbar
colormap(jet)
y_inds = 1:5:20;
y_values = round(new_thr(1:5:end));
yticks(y_inds);
yticklabels(y_values);
x_inds = 1:5:20;
x_values = round((new_slp(1:5:end))*100)/100;
xticks(x_inds);
xticklabels(x_values);
currentx = min(find(new_slp>=BCI_threshold(2)));
currenty = min(find(new_thr>=BCI_threshold(1)));
plot(currentx,currenty,'co','markersize',10)
xlabel('Upper threshold (FU)')
ylabel('Lower threshold (FL)')
cb = colorbar;
ylabel(cb,'Hit rate')
title(sprintf('Hit rate (integral threshold = %.2f V·s, T = %ds)', integral_hit, response_T))

% Interactive selection
a = ginput(1); a = round(a);
BCI_threshold = [new_thr(a(2)); new_slp(a(1))];
fprintf('Selected:  FL = %.4f,  FU = %.4f\n', BCI_threshold(1), BCI_threshold(2));
fprintf('Hit rate = %.0f%%,  median T_hit = %.1f s\n', ...
    hit_rate(a(2),a(1))*100, median_T(a(2),a(1)));

%% ---- Plot: signal with thresholds ----
figure(34566); clf
plot(valueHistory(:,cn)); hold on;
plot(xlim,[1 1]*BCI_threshold(1),'r','LineWidth',1.5);
plot(xlim,[1 1]*BCI_threshold(2),'b','LineWidth',1.5);
legend('signal','FL','FU')
ylabel('Fluorescence')
xlabel('Frame')
