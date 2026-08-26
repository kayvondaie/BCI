
function out = BCI_analog_display(in,varagin)
persistent prev_trial current_ind
global BCI_threshold BCI_params

f = @(x) (x - BCI_threshold(1)).*((x-BCI_threshold(1))>0).*(3.3/diff(BCI_threshold));
out = f(mean(in));
out(out>3.3) = 3.3;
out(out<0) = 0;

hSI = evalin('base','hSI');
base = hSI.hScan2D.logFileStem;

% ---- Initialize per-session tracking ----
if ~isfield(BCI_params, base) || ~isfield(BCI_params.(base), 'cn_trace')
    BCI_params.(base).cn_trace            = nan(1, 360000);
    BCI_params.(base).cn_output           = nan(1, 360000);
    BCI_params.(base).current_thresholds  = nan(2, 360000);
    BCI_params.(base).frames_per_trial    = [];
    BCI_params.(base).trial_frame_count   = 0;
    current_ind = 1;
end

trial_num = hSI.hScan2D.logFileCounter;
if trial_num > prev_trial
    hSICtl = evalin('base','hSICtl');
    str = [hSI.hScan2D.logFilePath,'\',hSI.hScan2D.logFileStem,'_threshold_',num2str(trial_num),'.mat'];

    rois = hSICtl.hGUIData.integrationRoiOutputChannelControlsV5.tbRoiSelection.Data;
    selected_rois = (cell2mat(rois(:,1)));

    save(str,'BCI_threshold','selected_rois');

    str = datestr(now);str(str==' ') = '_';
    str(str==':') = '_';
    str = [hSI.hScan2D.logFilePath,'\',hSI.hScan2D.logFileStem,'_threshold_',str,'.mat'];
    save(str,'BCI_threshold','selected_rois');

    % Record frames in completed trial
    if BCI_params.(base).trial_frame_count > 0
        BCI_params.(base).frames_per_trial(end+1) = BCI_params.(base).trial_frame_count;
        BCI_params.(base).trial_frame_count = 0;
    end

    bonsai_threshold_noplot;
end
prev_trial = trial_num;

% ---- Per-frame tracking ----
BCI_params.(base).cn_trace(current_ind)           = mean(in);
BCI_params.(base).cn_output(current_ind)          = out;
BCI_params.(base).current_thresholds(:,current_ind) = BCI_threshold;
BCI_params.(base).trial_frame_count               = BCI_params.(base).trial_frame_count + 1;
current_ind = current_ind + 1;

BCI_params.time = mod(BCI_params.time + 1,1000-1);
if BCI_params.time == 1
    BCI_params.line.YData  = BCI_params.line.YData*0;  BCI_params.stim_line.YData  = BCI_params.stim_line.YData*0;
    BCI_params.line2.YData = BCI_params.line.YData*0;  BCI_params.stim_line2.YData = BCI_params.stim_line.YData*0;
end
BCI_params.line.YData(BCI_params.time)  = out;
BCI_params.line2.YData(BCI_params.time) = mean(in);

end
