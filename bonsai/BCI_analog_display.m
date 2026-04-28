
function out = BCI_analog_display(in,varagin)
persistent prev_trial 
global BCI_threshold BCI_params

if isempty('prev_trial')
    prev_trial = 0;
end

f = @(x) (x - BCI_threshold(1)).*((x-BCI_threshold(1))>0).*(3.3/diff(BCI_threshold));
out = f(mean(in));
out(out>3.3) = 3.3;
out(out<0) = 0;
%disp(out)




hSI = evalin('base','hSI');
trial_num = hSI.hScan2D.logFileCounter;
if trial_num > prev_trial
    hSICtl = evalin('base','hSICtl');
    str = [hSI.hScan2D.logFilePath,'\',hSI.hScan2D.logFileStem,'_threshold_',num2str(trial_num),'.mat']
    
    rois = hSICtl.hGUIData.integrationRoiOutputChannelControlsV5.tbRoiSelection.Data;
    N = size(rois,1);
    selected_rois = (cell2mat(rois(:,1)));
    
    save(str,'BCI_threshold','selected_rois');
    
    str = datestr(now);str(str==' ') = '_';
    str(str==':') = '_';
    str = [hSI.hScan2D.logFilePath,'\',hSI.hScan2D.logFileStem,'_threshold_',str,'.mat'];
    save(str,'BCI_threshold','selected_rois');
end
prev_trial = trial_num;


BCI_params.time = mod(BCI_params.time + 1,1000-1);
if BCI_params.time == 1;
    BCI_params.line.YData = BCI_params.line.YData*0;BCI_params.stim_line.YData = BCI_params.stim_line.YData*0;
    BCI_params.line.YData = BCI_params.line.YData*0;BCI_params.stim_line2.YData = BCI_params.stim_line.YData*0;
end
BCI_params.line.YData(BCI_params.time) = (out);
BCI_params.line2.YData(BCI_params.time) = ((in));





%     end
end