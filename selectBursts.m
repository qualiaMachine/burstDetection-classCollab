clear all;    % Clear any workspace vars
close all;  % Close all figures (except those of imtool.)
%% Instructions
% 1. Start the program (hit Run)

% 2. Verify animalList variable above keyboard (input whichever animals
% you'd like to process -- be sure to communicate with team so we each 
% process different animals!)

% 3. Hit continue

% 4. You will then be asked for location of data directory (figsWithStatsDetect folder) as well as save
% directory.  If you quit the program (hit no at checkup point) previously,
% select the same folder you selected last time to continue your work.

% 5. 2x8 trial figures will begin to show up.  For each window, your cursor
% will change appearance, and you should select each burst in the figure with
% two points (bottom half and top half of burst).  Avoid selecting points too close
% to the y-axis limits of each subplot (trial).  We will average the x-axis
% location of each pair of points per burst in order to define the center
% point of the window to later be extracted (i.e. if the burst appears to start with cells
% firing towards the top/bottom of the cortical column, we take the average
% between start of firing in top of burst and bottom of burst as a good
% approximation of the center point of the burst).  

% 6. DON'T exit the program by any means other than selecting 'No' when the
% progam checks up on you after however many 2x8 trial figures you want it to check up
% on you repetitively (set var right below this to alter it).
fileCountCheckup = 30; % alter this to change how frequently you're asked to quit the program (numFiles to process before checkup)

% 7. DO NOT label bursts that are indicated as detected by stats method--that is unnecessary work.


%% Begin program
%% Animal ist to be processed
disp('Verify animal list in selectBursts.m and click continue.')
animalList = {'Gunter','Ice King'}; %% set the list of animal's whose data you'll inspect (also sets name of save file).
keyboard % to allow you to double check you animalList settings... is this correct for the data you wish to process?

%% Set location of mutliTrialPlots
startDir = ['12345678910111213141516171819'];
figDirName = 'figsWithStatsDetect';
while ~strcmp(startDir(end-length(figDirName)+1:end),figDirName)
    startDir = uigetdir(pwd,'Set location of figsWithStatsDetect folder');
    if ~strcmp(startDir(end-length(figDirName)+1:end),figDirName)
        disp('You must select the mutliTrialPlots folder')
    end
end
% startDir = 'T:\BurstDetection\figsWithStatsDetect';

%% Ask user where to save results
saveFolder = uigetdir(pwd,'Please select save folder. If you have previous results saved, you must use the same folder or move your file to continue your work.  Program will assume you have unfinished work if there is already a save file in selected folder.');
% saveFolder = 'T:\BurstDetection\figs\processedResults';

saveFile = [saveFolder filesep 'burstCoords_' strjoin(animalList) '.mat'];

%% Load in previous work if it exists
if exist(saveFile,'file')
    load(saveFile)
else
    animalIndStart = 1;
    dateIndStart = 1;
    condIndStart = 1;
    stimIndStart = 1;
    fileIndStart = 1;
end
filesProcessed = 0;
button = '';
for animalInd = animalIndStart:length(animalList)
    animal = animalList{animalInd};

    animalDir = [startDir filesep animal];
    dates = getFolders(animalDir);

    for dateInd = dateIndStart:length(dates)
        date = dates(dateInd).name;
        animalDateDir = [animalDir filesep date];
        conds = getFolders(animalDateDir);
        for condInd = condIndStart:length(conds)
            cond = conds(condInd).name;
            condDir = [animalDateDir filesep cond];
            stims = getFolders(condDir);
            for iStim = stimIndStart:length(stims)
                stim = stims(iStim).name;
                stimDir = [condDir filesep stim];
                %% Load each multiTrial plot
                coordDataFileList = dir([stimDir filesep '*.fig']);
                for iFile = fileIndStart:length(coordDataFileList)
                    if strcmp(button,'No')
                        fileIndStart = iFile;
                        animalIndStart = animalInd;
                        dateIndStart = dateInd;
                        condIndStart = condInd;
                        stimIndStart = iStim;

                        save(saveFile,'animalIndStart','dateIndStart','condIndStart','stimIndStart','fileIndStart','coordDataFileList')
                        return
                    end
                    [burstLocC,burstLocR] = gatherBurstCoordinates(stimDir,coordDataFileList(iFile).name);
                    % store file's burst coordinates
                    coordDataFileList(iFile).burstLocC = burstLocC;
                    coordDataFileList(iFile).burstLocR = burstLocR;
                    filesProcessed = filesProcessed + 1;
                    % check on user
                    if filesProcessed == fileCountCheckup
                        button = questdlg(['Would you like to continue with another ' num2str(fileCountCheckup) ' more files?']);
                        filesProcessed = 0;
                    end

                end
            end
            
        end
            
    end
end
disp('Congrats!  You''ve finished processing your specified list of animal data.')
