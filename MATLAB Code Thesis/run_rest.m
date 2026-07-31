clear,clc

% Script to run rest triggers
clear

rest_number = 1; % rest number 1, 2, or 3

Lstim = 60; % length of rest in sec
trigger_val = 20+rest_number; % trigger value
trig_isi = 1; % interval between triggers
fs = 48000; % 48000 sampling rate

% triggers:
trig_dur = round(1e-3*fs); % trigger duration in samples
trig_ons = 1:fs*trig_isi:Lstim*fs; % onset of each stimulation block
trigger = zeros(Lstim*fs,1);
for kk = 1:length(trig_ons)
    trigger(trig_ons(kk):trig_ons(kk)+trig_dur+1,1) = 1; 
end
stim = [zeros(Lstim*fs,3) trigger];

%% init psychportaudio
try
    PsychPortAudio('GetOpenDeviceCount')
    PsychPortAudio('close');
catch
end
InitializePsychSound;
dev = PsychPortAudio('GetDevices');

devid = find_ASIO_devID; 
selectchannel =  [4 5 6 17;0 0 0 0];
nchans =  size(selectchannel,2);
pah = PsychPortAudio('Open', devid, [], 0, fs, nchans, [], [], selectchannel);
pa_status = PsychPortAudio('GetStatus',pah);
deviceId = pa_status.OutDeviceIndex;
PsychPortAudio('Volume',pah,0);
pa_status = PsychPortAudio('GetStatus',pah);
deviceId = pa_status.OutDeviceIndex;
dev(deviceId)
PsychPortAudio('FillBuffer', pah, [stim]'); % check that nchans is correct by filling buffer
PsychPortAudio('Start', pah, 1, 0, 0, GetSecs+.1);
PsychPortAudio('Stop', pah, 1);
PsychPortAudio('Volume',pah,1);


%% RUN STIMULATION with psychtoolbox

% init triggerbox
trig = HEATriggerbox();
trig.find_triggerbox_win();
trig.connect();
if trig.is_connected()
   trig.set_trigger(trigger_val);
end

KbName('UnifyKeyNames');
escKey = KbName('ESCAPE');
commandwindow;
while KbCheck; end
WaitSecs(.1);
exptime = GetSecs;
fprintf('\n\nPress to begin rest recording\n')
fprintf('Press ESC to exit\n\n')
KbWait(-1);

block_idx = 1;

try  
    % ListenChar(2);
    breakflag = 0;

    offset = .5;
    fprintf('Press to begin rest recording\n')

    KbWait(-1);

    tnow = GetSecs;
    PsychPortAudio('Start', pah, 1, tnow+offset);
    s = PsychPortAudio('GetStatus', pah);
    t_audio = GetSecs;
    if ~s.Active
        while 1 % wait untill audio starts
            s = PsychPortAudio('GetStatus', pah);
            [~,~,keyCode] = KbCheck(-1);
            if s.Active
                t_audio = GetSecs;
                break;
            end
            if keyCode(escKey)
                fprintf('escape\n')
                breakflag=1;
                PsychPortAudio('Stop', pah, 0);
                break;
            end
        end
    end
    fprintf('Playing\n')
    s = PsychPortAudio('GetStatus', pah);
    if s.Active
        dat.starttime = GetSecs-exptime;
        while 1 % wait untill audio stops
            s = PsychPortAudio('GetStatus', pah);

            [~,~,keyCode] = KbCheck(-1);
            if ~s.Active
                break
            end
            if keyCode(escKey)
                fprintf('Escape\n')
                breakflag=1;
                PsychPortAudio('Stop', pah, 0);
                break
            end
        end
    end

    PsychPortAudio('Stop', pah, 1);

    fprintf('End of rest\n\n\n')
    
    
catch
    ListenChar(0);
    psychrethrow(psychlasterror);
    PsychPortAudio('Stop', pah);
end
ListenChar(0);
PsychPortAudio('Stop', pah);
trig.disconnect();
delete(trig);