function varargout = Photostim_BCI_gui4(varargin)
% PHOTOSTIM_BCI_GUI4 MATLAB code for Photostim_BCI_gui4.fig
%      PHOTOSTIM_BCI_GUI4, by itself, creates a new PHOTOSTIM_BCI_GUI4 or raises the existing
%      singleton*.
%
%      H = PHOTOSTIM_BCI_GUI4 returns the handle to a new PHOTOSTIM_BCI_GUI4 or the handle to
%      the existing singleton*.
%
%      PHOTOSTIM_BCI_GUI4('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PHOTOSTIM_BCI_GUI4.M with the given input arguments.
%
%      PHOTOSTIM_BCI_GUI4('Property','Value',...) creates a new PHOTOSTIM_BCI_GUI4 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Photostim_BCI_gui4_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Photostim_BCI_gui4_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Photostim_BCI_gui4

% Last Modified by GUIDE v2.5 26-Feb-2025 15:51:26

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Photostim_BCI_gui4_OpeningFcn, ...
                   'gui_OutputFcn',  @Photostim_BCI_gui4_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Photostim_BCI_gui4 is made visible.
function Photostim_BCI_gui4_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Photostim_BCI_gui4 (see VARARGIN)

% Choose default command line output for Photostim_BCI_gui4
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
global BCI_params
assignin('base','hPhotostim_BCI',handles)
% assignin('base','BCI_params',BCI_params)
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');
try
    handles.lower_1_edit.String = num2str(BCI_params.thr_low(1));
    handles.upper_1_edit.String = num2str(BCI_params.thr_high(1));
catch
%     BCI_params.thr_low = [0 ];
%     BCI_params.thr_high = [10000];
end

BCI_params.multiplier = str2num(handles.multiplier_edit.String);
BCI_params.(base).stim_time = nan(1,10000);
BCI_params.(base).stim_time(1) = 0;
BCI_params.(base).stim_group = nan(1,10000);
BCI_params.executed_stims = 0;
BCI_params.tau = str2num(handles.tau_edit.String);
BCI_params.hit_group = eval(handles.hit_group_edit.String);
BCI_params.stim_int = eval(handles.stim_int_edit.String);
BCI_params.low_activity = 0;
BCI_params.stim_delay = rand*(diff(BCI_params.stim_int)) + BCI_params.stim_int(1);

val = handles.gate_fcn_popup.Value;
if val == 2;
    BCI_params.gate_fcn = @(x,tau) exp(-x/tau);
elseif val == 1
    BCI_params.gate_fcn = @(x,tau) (x < tau);
end

hSICtl.hGUIData.integrationRoiOutputChannelControlsV5.pmOutputChannel.Value = 1;
% hSI.hIntegrationRoiManager.outputChannelsFunctions{1} = '@(vals,varargin)BCI_analog_gated(vals)';
% hSI.hIntegrationRoiManager.outputChannelsFunctions{1} = '@(vals,varargin)BCI_analog2_gated_trigger_stim(vals)';
hSI.hIntegrationRoiManager.outputChannelsFunctions{1} = '@(vals,varargin)BCI_analog2_gated_delta_trigger_stim(vals)';
hSI.hPhotostim.laserActiveSignalAdvance = 0;

BCI_params.out = cell(1,1);
try
%     hSI.hScanners{2}.xGalvo.feedbackVoltFcn = @(x) x;
%     hSI.hScanners{2}.yGalvo.feedbackVoltFcn = @(x) x;
    hSI.hScanners{2}.xGalvo.feedbackVoltLUT = zeros(1,2);
    hSI.hScanners{2}.yGalvo.feedbackVoltLUT = zeros(1,2);
    hSI.hPhotostim.logging = 1;
end

BCI_params.fig = figure(999);clf
set(gcf,'position',[847    49   349   451])
BCI_params.ax = subplot(211);
BCI_params.len = 1000;
BCI_params.time = 1;
BCI_params.stim_line = plot(zeros(1,BCI_params.len));hold on;
BCI_params.line = plot(zeros(1,BCI_params.len));hold on;
xlim([0 BCI_params.len]);

BCI_params.ax2 = subplot(212);
BCI_params.stim_line2 = plot(zeros(1,BCI_params.len));hold on;
BCI_params.line2 = plot(zeros(1,BCI_params.len));hold on;

BCI_params.baseline = ones(1,10)*100;
% ylim([-.5 3.4]);
axis tight
xlim([0 BCI_params.len]);


% for i = 1:2
%     hRoi = scanimage.mroi.Roi();
%     hIntegrationField = scanimage.mroi.scanfield.fields.IntegrationField();
%     z = hSICtl.hModel.hFastZ.positionTarget;
%     hRoi.add(z,hIntegrationField);
%     if i ==1
%         hIntegrationField.centerXY = [1 1];
%     else
%         hIntegrationField.centerXY = [1 -1];
%     end
%     hIntegrationField.sizeXY = [1 1];
%     
%     [xx,yy] = ndgrid(-1:0.01:1,-1:0.01:1);
%     rr = sqrt(xx.^2 + yy.^2);
%     mask = double( rr<1 );
%     hIntegrationField.mask = mask;
%     
%     hSI.hIntegrationRoiManager.roiGroup.add(hRoi);
%     hSI.hIntegrationRoiManager.roiGroup.rois(end).name = ['ConditionedNeuron',num2str(i)];
% end

% UIWAIT makes Photostim_BCI_gui4 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Photostim_BCI_gui4_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function lower_1_edit_Callback(hObject, eventdata, handles)
% hObject    handle to lower_1_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params BCI_threshold
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');
BCI_params.thr_low(1) = str2double(get(hObject,'String'));
BCI_threshold(1) = str2double(get(hObject,'String'))
% Hints: get(hObject,'String') returns contents of lower_1_edit as text
%        str2double(get(hObject,'String')) returns contents of lower_1_edit as a double


% --- Executes during object creation, after setting all properties.
function lower_1_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lower_1_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function upper_1_edit_Callback(hObject, eventdata, handles)
% hObject    handle to upper_1_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params BCI_threshold
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');
BCI_params.thr_high(1) = str2double(get(hObject,'String'));
x = 1:1000;
handles.Stim_1_ax.Children(2).XData = x;
handles.Stim_1_ax.Children(2).YData = repmat(BCI_params.thr_high(1),1,length(x));
BCI_threshold(2) = str2double(get(hObject,'String'))

% Hints: get(hObject,'String') returns contents of upper_1_edit as text
%        str2double(get(hObject,'String')) returns contents of upper_1_edit as a double


% --- Executes during object creation, after setting all properties.
function upper_1_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to upper_1_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function lower_2_edit_Callback(hObject, eventdata, handles)
% hObject    handle to lower_2_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');

BCI_params.thr_low(2) = str2double(get(hObject,'String'));
x = 1:1000;
handles.Cond_2_ax.Children(2).XData = x;
handles.Cond_2_ax.Children(2).YData = repmat(BCI_params.thr_low(2),1,length(x));
% Hints: get(hObject,'String') returns contents of lower_2_edit as text
%        str2double(get(hObject,'String')) returns contents of lower_2_edit as a double


% --- Executes during object creation, after setting all properties.
function lower_2_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lower_2_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function upper_2_edit_Callback(hObject, eventdata, handles)
% hObject    handle to upper_2_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');
BCI_params.thr_high(2) = str2double(get(hObject,'String'));
x = 1:1000;
handles.Stim_2_ax.Children(2).XData = x;
handles.Stim_2_ax.Children(2).YData = repmat(BCI_params.thr_high(2),1,length(x));
% Hints: get(hObject,'String') returns contents of upper_2_edit as text
%        str2double(get(hObject,'String')) returns contents of upper_2_edit as a double


% --- Executes during object creation, after setting all properties.
function upper_2_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to upper_2_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function neuron_1_prob_edit_Callback(hObject, eventdata, handles)
% hObject    handle to neuron_1_prob_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');
BCI_params.neuron_1_prob = str2double(get(hObject,'String'));
% Hints: get(hObject,'String') returns contents of neuron_1_prob_edit as text
%        str2double(get(hObject,'String')) returns contents of neuron_1_prob_edit as a double


% --- Executes during object creation, after setting all properties.
function neuron_1_prob_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to neuron_1_prob_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in reset_thr_button.
function reset_thr_button_Callback(hObject, eventdata, handles)
% hObject    handle to reset_thr_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params BCI_threshold
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');
[~,valueHistory] = hSI.hIntegrationRoiManager.getIntegrationHistory();
cn = find([hSICtl.hGUIData.integrationRoiOutputChannelControlsV5.tbRoiSelection.Data{:,1}]==1);
X = valueHistory(:,cn);
thr = zeros(1,size(X,2));
for i = 1:size(X,2);
    a = X(:,i);
    thr(i) = prctile(a(a~=0),str2num(handles.percentile_edit.String));
end
BCI_params.thr_low = thr;
BCI_params.thr_high = prctile(X,100);
BCI_threshold = [BCI_params.thr_low;BCI_params.thr_high];
% BCI_threshold(2,:) = BCI_threshold(2,:) + BCI_threshold(1,:);
handles.lower_1_edit.String = num2str(BCI_params.thr_low);
handles.upper_1_edit.String = num2str(BCI_params.thr_high);

assignin('base','hPhotostim_BCI',handles)
assignin('base','BCI_params',BCI_params)
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');
try
    handles.lower_1_edit.String = num2str(BCI_params.thr_low(1));
    handles.upper_1_edit.String = num2str(BCI_params.thr_high(1));
catch
    BCI_params.thr_low = [0 ];
    BCI_params.thr_high = [10000];
end

hSICtl = evalin('base','hSICtl');
trial_num = hSI.hScan2D.logFileCounter;

str = [hSI.hScan2D.logFilePath,'\',hSI.hScan2D.logFileStem,'_threshold_',num2str(trial_num),'.mat']

rois = hSICtl.hGUIData.integrationRoiOutputChannelControlsV5.tbRoiSelection.Data;
N = size(rois,1);
selected_rois = (cell2mat(rois(:,1)));
if N > 1 & strcmp(hSI.hIntegrationRoiManager.outputChannelsFunctions{1},'@(vals,varargin)BCI_analog_display_N_neurons(vals)')
    global manifold_thr vectors
    manifold_thr = calculate_manifold_thr(vectors,[50,100])
    % manifold_thr = prctile(dff,percentiles)
end
save(str,'BCI_threshold','selected_rois');

str = datestr(now);str(str==' ') = '_';
str(str==':') = '_';
str = [hSI.hScan2D.logFilePath,'\',hSI.hScan2D.logFileStem,'_threshold_',str,'.mat'];
save(str,'BCI_threshold','selected_rois');


% --- Executes on button press in dummy_button.
function dummy_button_Callback(hObject, eventdata, handles)
% hObject    handle to dummy_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Dummy_integration_rois_on_photostim


function SLM_num_edit_Callback(hObject, eventdata, handles)
% hObject    handle to SLM_num_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
for i = 1:length(hSI.hPhotostim.stimRoiGroups)
    hSICtl.hGUIData.photostimControlsV5.lbStimGroups.Value = length(hSI.hPhotostim.stimRoiGroups);
    hSICtl.remStimGroup();
end
Dummy_integration_rois_on_photostim

% Hints: get(hObject,'String') returns contents of SLM_num_edit as text
%        str2double(get(hObject,'String')) returns contents of SLM_num_edit as a double


% --- Executes during object creation, after setting all properties.
function SLM_num_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SLM_num_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in slm_to_int_button.
function slm_to_int_button_Callback(hObject, eventdata, handles)
% hObject    handle to slm_to_int_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
SLM_to_integration


function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params
clearvars -global BCI_params
% Hint: delete(hObject) closes the figure
delete(hObject);



function tau_edit_Callback(hObject, eventdata, handles)
% hObject    handle to tau_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params
BCI_params.tau = str2num(handles.tau_edit.String);
% Hints: get(hObject,'String') returns contents of tau_edit as text
%        str2double(get(hObject,'String')) returns contents of tau_edit as a double


% --- Executes during object creation, after setting all properties.
function tau_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tau_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in AO_fcn_popup.
function AO_fcn_popup_Callback(hObject, eventdata, handles)
% hObject    handle to AO_fcn_popup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');

str_fcn = handles.AO_fcn_popup.String{handles.AO_fcn_popup.Value};
str = ['@(vals,varargin)',str_fcn,'(vals)'];
hSI.hIntegrationRoiManager.outputChannelsFunctions{1} = str;
% Hints: contents = cellstr(get(hObject,'String')) returns AO_fcn_popup contents as cell array
%        contents{get(hObject,'Value')} returns selected item from AO_fcn_popup


% --- Executes during object creation, after setting all properties.
function AO_fcn_popup_CreateFcn(hObject, eventdata, handles)
% hObject    handle to AO_fcn_popup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Name_ROIs_button.
function Name_ROIs_button_Callback(hObject, eventdata, handles)
% hObject    handle to Name_ROIs_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
hSI = evalin('base','hSI');
for i = 1:length(  hSI.hIntegrationRoiManager.roiGroup.rois)
    hSI.hIntegrationRoiManager.roiGroup.rois(i).name = ['ConditionedNeuron',num2str(i)];
end

% hSI.hScanners{2}.xGalvo.feedbackVoltFcn = @(x) x;
% hSI.hScanners{2}.yGalvo.feedbackVoltFcn = @(x) x;
% hSI.hPhotostim.logging = 1;



function offset_edit_Callback(hObject, eventdata, handles)
% hObject    handle to offset_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');
BCI_params.offset = str2num(handles.offset_edit.String);
% Hints: get(hObject,'String') returns contents of offset_edit as text
%        str2double(get(hObject,'String')) returns contents of offset_edit as a double


% --- Executes during object creation, after setting all properties.
function offset_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to offset_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in update_button.
function update_button_Callback(hObject, eventdata, handles)
% hObject    handle to update_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');
BCI_params.tau = str2num(handles.tau_edit.String);

if handles.AO_fcn_popup.Value == 1;
    BCI_params.fcn = @(x,tau,offset) x*(x<=tau);
elseif handles.AO_fcn_popup.Value == 2
    BCI_params.fcn = @(x,tau,offset) (1-offset)*exp(-x/tau)+offset;
end

BCI_params.offset = str2num(handles.offset_edit.String);
BCI_params.thr_low(1) = str2num(handles.lower_1_edit.String);
BCI_params.thr_high(1) = str2num(handles.upper_1_edit.String);
BCI_params.out = cell(1,1);
try
    hSICtl.hGUIData.integrationRoiOutputChannelControlsV5.pmOutputChannel.Value = 2;
    hSI.hIntegrationRoiManager.outputChannelsFunctions{2} = '@(vals,varargin)BCI_analog_gated(vals)';
    hSI.hPhotostim.laserActiveSignalAdvance = 0;
    hSI.hScanners{2}.xGalvo.feedbackVoltFcn = @(x) x;
    hSI.hScanners{2}.yGalvo.feedbackVoltFcn = @(x) x;
    hSI.hPhotostim.logging = 1;
end


function hit_group_edit_Callback(hObject, eventdata, handles)
% hObject    handle to hit_group_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params
BCI_params.hit_group = eval(handles.hit_group_edit.String);
% Hints: get(hObject,'String') returns contents of hit_group_edit as text
%        str2double(get(hObject,'String')) returns contents of hit_group_edit as a double


% --- Executes during object creation, after setting all properties.
function hit_group_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hit_group_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function percentile_edit_Callback(hObject, eventdata, handles)
% hObject    handle to percentile_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of percentile_edit as text
%        str2double(get(hObject,'String')) returns contents of percentile_edit as a double


% --- Executes during object creation, after setting all properties.
function percentile_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to percentile_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function multiplier_edit_Callback(hObject, eventdata, handles)
% hObject    handle to multiplier_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of multiplier_edit as text
%        str2double(get(hObject,'String')) returns contents of multiplier_edit as a double
global BCI_params;
BCI_params.multiplier = str2num(handles.multiplier_edit.String);

% --- Executes during object creation, after setting all properties.
function multiplier_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to multiplier_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in gate_fcn_popup.
function gate_fcn_popup_Callback(hObject, eventdata, handles)
% hObject    handle to gate_fcn_popup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns gate_fcn_popup contents as cell array
%        contents{get(hObject,'Value')} returns selected item from gate_fcn_popup

global BCI_params
val = handles.gate_fcn_popup.Value;
if val == 2;
    BCI_params.gate_fcn = @(x,tau) exp(-x/tau);
elseif val == 1
    BCI_params.gate_fcn = @(x,tau) (x < tau);
end
% --- Executes during object creation, after setting all properties.
function gate_fcn_popup_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gate_fcn_popup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function stim_int_edit_Callback(hObject, eventdata, handles)
% hObject    handle to stim_int_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params;
BCI_params.stim_int = eval(handles.stim_int_edit.String);
% Hints: get(hObject,'String') returns contents of stim_int_edit as text
%        str2double(get(hObject,'String')) returns contents of stim_int_edit as a double


% --- Executes during object creation, after setting all properties.
function stim_int_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stim_int_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Dthr_button.
function Dthr_button_Callback(hObject, eventdata, handles)
% hObject    handle to Dthr_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params
delta_threshold_calculate


% --- Executes on button press in spont_thresholds_buttom.
function spont_thresholds_buttom_Callback(hObject, eventdata, handles)
% hObject    handle to spont_thresholds_buttom (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params
global BCI_threshold
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');
spont_thresholds2
thr = BCI_threshold

BCI_params.thr_low = thr(1);
BCI_params.thr_high = thr(2);
BCI_threshold = [BCI_params.thr_low;BCI_params.thr_high];
% BCI_threshold(2,:) = BCI_threshold(2,:) + BCI_threshold(1,:);
handles.lower_1_edit.String = num2str(BCI_params.thr_low);
handles.upper_1_edit.String = num2str(BCI_params.thr_high);

str = [hSI.hScan2D.logFilePath,'\',hSI.hScan2D.logFileStem,'spont_threshold_map_','trial_',num2str(hSI.hScan2D.logFileCounter),'.png'];
saveas(figure(9495), str);
str = [hSI.hScan2D.logFilePath,'\',hSI.hScan2D.logFileStem,'spont_threshold_plot_trial_',num2str(hSI.hScan2D.logFileCounter),'.png'];
saveas(figure(34566), str);

% --- Executes on button press in bpod_threshold.
function bpod_threshold_Callback(hObject, eventdata, handles)
% hObject    handle to bpod_threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global BCI_params
global BCI_threshold
hSI = evalin('base','hSI');base = hSI.hScan2D.logFileStem;hSICtl = evalin('base','hSICtl');
use_last_10_trials = true;

% ---- Find most recent Bonsai session on Z:\ ----
bonsai_root = 'Z:\';
a = dir(bonsai_root);
a = a(3:end);  % skip . and ..
a = a([a.isdir]);  % only folders
folders = sort({a.name}');
bonsai_session = fullfile(bonsai_root, folders{end}, 'behavior');
fprintf('Bonsai session: %s\n', bonsai_session);

% ---- Parse ResponsePeriod.json (go cue times) ----
fid = fopen(fullfile(bonsai_session, 'SoftwareEvents', 'ResponsePeriod.json'), 'r');
go_times = [];
while ~feof(fid)
    line = fgetl(fid);
    if isempty(strtrim(line)); continue; end
    ev = jsondecode(line);
    go_times(end+1) = ev.timestamp;
end
fclose(fid);

% ---- Parse SpoutPosition.csv to get reward times ----
sp = readtable(fullfile(bonsai_session, 'OperationControl', 'SpoutPosition.csv'));
sp.Properties.VariableNames = strtrim(sp.Properties.VariableNames);
t_sp = sp.Seconds;
pos = sp.Value;
P_max = max(pos);

% ---- Compute time-to-reward per trial ----
tct = nan(1, length(go_times));
for i = 1:length(go_times)
    t_go = go_times(i);
    % Next trial start (or end of data)
    if i < length(go_times)
        t_end = go_times(i+1);
    else
        t_end = t_sp(end);
    end
    % Find first time spout reaches P_max within this trial
    hit_idx = find(t_sp >= t_go & t_sp <= t_end & pos >= P_max - 0.1, 1);
    if ~isempty(hit_idx)
        tct(i) = t_sp(hit_idx) - t_go;
    end
end

% ---- Multiplier sweep (same logic as bpod version) ----
tctt = 1:length(tct);
tct_first = tct(1:min(10, length(tct)));
tctt_first = tctt(1:min(10, length(tct)));
if use_last_10_trials && length(tct) > 10
    tct = tct(end-9:end);
    tctt = tctt(end-9:end);
end

multipliers = linspace(1, 3, 500);
num = '1';
while true
    str = [hSI.hScan2D.logFilePath, '\', hSI.hScan2D.logFileStem, '_threshold_', num, '.mat'];
    if exist(str, 'file')
        old_thr = load(str, 'BCI_threshold');
        old_thr = old_thr.BCI_threshold;
        break;
    else
        num = num2str(str2double(num) + 1);
    end
end

multipliers_old = multipliers * diff(BCI_threshold) / diff(old_thr);
clear hit_rate mean_time_to_hit hit_rate_first mean_time_to_hit_first
for i = 1:length(multipliers)
    multiplier = multipliers(i);
    hit_rate(i) = mean(tct * multiplier < 10);
    mean_time_to_hit(i) = nanmean(tct * multiplier);

    hit_rate_first(i) = mean(tct_first * multipliers_old(i) < 10);
    mean_time_to_hit_first(i) = nanmean(tct_first * multipliers_old(i));
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

str = [hSI.hScan2D.logFilePath,'\',hSI.hScan2D.logFileStem,'_threshold_calc_',num2str(hSI.hScan2D.logFileCounter),'.png'];
saveas(figure(1338), str);


% figure(99945)
% title('add water!!!!!')
