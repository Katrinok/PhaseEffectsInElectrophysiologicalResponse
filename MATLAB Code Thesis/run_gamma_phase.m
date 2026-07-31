clear,clc

% Script to run AV gamma stimulation with phase offsets

dat.subid = 'SA4'; % subject identifier (for saving)

% Trials: (eg. 1 sec baseline, 1.5 sec stimulation)
stimrate = 40; % in hz 
fs = 48000; % 48000 sampling rate
L = 1.5; % length of stimulus block = 1 sec
Lstim = 2.5; % length of stimulus block including silence
phi = [0 pi/4 pi/2 3*pi/4 pi]; % phase shift values to run in radians
dNs = (phi./(2*pi)) .* (fs/stimrate); % phase shifts in samples
Nphi  = length(phi); % how many phase shifts?
Nreps = 120; % how many stim per phi
Ntrials = Nphi*Nreps; % how many blocks


click_len = [50 300]; % length of clicks for A and V in samples
gain = [.006 .5]; % A - V  % Audio gain 0.08: 70dB, continuoously, peak to peak 90dB ; old: 08->84dB; .12->90dB; .14->91dB .18->94dB

conds = zeros(1, Ntrials); % conditions 1 .. N (different shifts) - preallocation
for i = 1:Nreps
    idx = (i-1)*Nphi + (1:Nphi);
    conds(idx) = randperm(Nphi);
end

Ltot = Lstim*Ntrials; % total length of stimulation block
fprintf('Total stimulation time per block: %d secs\n',Ltot)
fprintf('No of blocks: %d\n',Ntrials)

try
catch 
    psychrethrow(psychlasterror); 
end

% collect experiment information in struct to be saved
dat.conditions = conds;
dat.phis = phi;
dat.click_len = click_len;
dat.starttimes = [];
dat.gain = gain;
dat.date = datestr(now);
dat.stim_dur = L;
dat.block_dur = Lstim;

%% make stimuli

t = linspace(0,Lstim-1/fs,Lstim*fs); % time one block
aclick = ones(click_len(1),1); % generate one A click
vclick = ones(click_len(2),1); % generate one V click
ons = 1:fs/stimrate:L*fs; % click onsets idx within block in samples

if max(click_len)*stimrate>L*fs || max(click_len)>max(diff(ons))
    error('click too long')
end
if length(unique(diff(ons)))>1 % distance between clicks should be constant
    warning('click rate does not match fs')
end

ya = zeros(Lstim*fs,1); % initialize one block of non-shifting (auditory)
for ii = 1:length(ons)
    ya(ons(ii):ons(ii)+length(aclick)-1,1) = aclick*gain(1);
    % alternatively AM modulated noise
end

yv = cell(1,Nphi); % phase shifted click trains (visual clicks)
for i = 1:Nphi
    dN = (phi(i)/(2*pi)) * (fs/stimrate);   % phase shift in samples
    ons_phi = 1 + dN : fs/stimrate : L*fs + dN;  % click onsets in samples
    ons_phi = round(ons_phi);

    yv{i} = zeros(Lstim*fs,1);
        
    for ii = 1:length(ons_phi)
        yv{i}(ons_phi(ii):ons_phi(ii)+length(vclick)-1,1) = vclick*gain(2);
    end
end

% make stimulus for all trials (A and V):
NsBlock = Lstim * fs;

stim = zeros(NsBlock*Ntrials, 3);

for i = 1:Ntrials
    idx = (i-1)*NsBlock + (1:NsBlock);
    stim(idx,:) = [ya ya yv{conds(i)}];
end

if Ltot~=(size(stim,1)/fs)
    error('time mismatch')
end

% triggers
trig_dur = round(1e-3*fs); % trigger duration in samples
trig_vals = 10:10:Nphi*10;
trigger = zeros(Lstim*fs,1);
trigger(1:trig_dur+1,1) = 1; 

block_ons = 1:fs*Lstim:Ltot*fs; % onset of each stimulation block

%subplot 211
%plot(stim(1:fs,:),'linewidth',2), ylim([0 0.01]), legend('auditory','visual')
%subplot 212
%plot(stim(1:fs*100,:),'linewidth',2)

%% Check the phis visually - Plots the stimulus for each phi

figure('Color','w','Name','Stimulus matrix check by phase');

plotDur = 0.15; % length of the trial to show in seconds
Ns = round(plotDur*fs);
tplot = (0:Ns-1)/fs;

% Scale auditory so it is visible next to visual
A_scale = gain(2)/gain(1);      % makes auditory peak comparable to visual
num = 1;
for p = 1:Nphi

    % Find the first trial in the randomized stimulus matrix with this phi
    trial_idx = find(conds == p, 1, 'first');

    % Samples belonging to that trial inside the full stimulus matrix
    idx0 = (trial_idx-1)*Lstim*fs + 1;
    idx = idx0 : idx0 + Ns - 1;

    auditory = stim(idx,1) * A_scale;
    visual   = stim(idx,3);

    subplot(Nphi,1,p)

    stairs(tplot, auditory, 'LineWidth', 1.4)
    hold on
    stairs(tplot, visual, 'LineWidth', 1.4)

    % Mark auditory click onsets
    aud_onsets = tplot(auditory > 0);
    vis_onsets = tplot(visual > 0);

    ylim([-0.05 0.65])
    xlim([0 plotDur])
    grid on

    title(sprintf('\\phi_{%.0f} = %.2f rad = %.0f°', num, phi(p), rad2deg(phi(p))), 'FontSize', 10, 'FontWeight','bold')

    ylabel('Amplitude')
    num = num+1;
    if p == 1
        legend(sprintf('Auditory × %.1f', A_scale), 'Visual')
    end
end

xlabel('Time [s]')
sgtitle('Auditory and visual stimulus for each phase condition')
%% init psychportaudio
try
    PsychPortAudio('GetOpenDeviceCount')
    PsychPortAudio('close');
catch
end
InitializePsychSound;
dev = PsychPortAudio('GetDevices');

devid = find_ASIO_devID; % 0 for PHYS2 HEAAUD
selectchannel =  [4 5 6 17;0 0 0 0]; %  [4 12;0 0]; ER2 + adat3
nchans =  size(selectchannel,2);
pah = PsychPortAudio('Open', devid, [], 0, fs, nchans, [], [], selectchannel);
pa_status = PsychPortAudio('GetStatus',pah);
deviceId = pa_status.OutDeviceIndex;
PsychPortAudio('Volume',pah,0);
pa_status = PsychPortAudio('GetStatus',pah);
deviceId = pa_status.OutDeviceIndex;
dev(deviceId)
PsychPortAudio('FillBuffer', pah, [stim repmat(trigger,Ntrials,1)]'); % check that nchans is correct by filling buffer
PsychPortAudio('Start', pah, 1, 0, 0, GetSecs+.1);
PsychPortAudio('Stop', pah, 1);
PsychPortAudio('Volume',pah,1);


%% RUN STIMULATION with psychtoolbox

% init triggerbox
trig = HEATriggerbox();
trig.find_triggerbox_win();
trig.connect();
if trig.is_connected()
   trig.set_trigger(trig_vals(conds(1)));
end

KbName('UnifyKeyNames');
escKey = KbName('ESCAPE');
commandwindow;
while KbCheck; end
WaitSecs(.1);
exptime = GetSecs;
fprintf('\n\nPress to begin experiment\n')
fprintf('Press ESC to exit\n\n')
KbWait(-1);

block_idx = 1;

try  
    % ListenChar(2);
    breakflag = 0;

    offset = .5;
    fprintf('Press to begin stimulation\n')

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
            if block_idx < Ntrials && s.ElapsedOutSamples > (block_ons(block_idx) + fs*.2) % if we have passed a block onset time
                block_idx=block_idx + 1;
                trig.set_trigger(trig_vals(conds(block_idx))); % set new trigger value
            end

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

    fprintf('End of stimulation\n\n\n')
    
    
catch
    ListenChar(0);
    psychrethrow(psychlasterror);
    PsychPortAudio('Stop', pah);
end
ListenChar(0);
PsychPortAudio('Stop', pah);
trig.disconnect();
delete(trig);
save(['./_data/gammaphase_',dat.subid,'.mat'],'dat','-v7.3');

