close all;
clear all;
windowSize = .35; % 350 msec --> make sure this is identical to stats-identified window size in createStatsBurstExamplesBatch!
trialXlimits = [-.5,1]; % sec (-.5 sec pre-stim to 1 sec)
singleTrialRasterImagesDir = 'T:\BurstDetection\figsWithStatsDetect\parsedStatsExamples';
imageFormat = '.jpg';
% savedDataDir = '/Users/tammi/Desktop/DeepANN/git/burstDetection/processedResults';
% savedDataDir = 'T:\BurstDetection\gitRepo\burstDetection-master\processedResults';
savedDataDir = 'T:\BurstDetection\figsWithStatsDetect\newSaveResults';

% animal = ' Resh';
animal = 'Delta';

load([savedDataDir filesep 'burstCoords_' animal '.mat'])

for iFile = 1:size(coordDataFileList,1)
    %% if no bursts selected on file, skip
    if isempty(coordDataFileList(iFile).burstLocC)
        continue
    end
    
    %% get multitrial raster plot name in order to get trial range
    multiTrialRasterPlotFile = coordDataFileList(iFile).name;
    beginTrialNumStartInd = strfind(multiTrialRasterPlotFile,'trials')+6;
    beginTrialNumEndInd = strfind(multiTrialRasterPlotFile,'t');
    beginTrialNumEndInd = beginTrialNumEndInd(beginTrialNumEndInd > beginTrialNumStartInd)-1;
    if length(beginTrialNumEndInd) > 1 || isempty(beginTrialNumEndInd)
        keyboard
    end
    startTrial = str2double(multiTrialRasterPlotFile(beginTrialNumStartInd:beginTrialNumEndInd));

    endTrialNumStartInd = beginTrialNumEndInd + 2;
    endTrialNumEndInd = strfind(multiTrialRasterPlotFile,'.fig')-1;
    endTrial = str2double(multiTrialRasterPlotFile(endTrialNumStartInd:endTrialNumEndInd));
    
    % This code relies on 16 rasters per multi-raster plot in order to
    % locate individual trials/burst locations
    if endTrial - startTrial ~= 15
        keyboard
    end
    
    burstLocCs = coordDataFileList(iFile).burstLocC;
    burstLocRs = coordDataFileList(iFile).burstLocR;
    
    %% remove out of bounds points
    % column limits
    leftColLim = -2.1442;
    rightColLim = 1.002;
    burstLocCs = burstLocCs(burstLocCs>leftColLim & burstLocCs<rightColLim);
    burstLocRs = burstLocRs(burstLocCs>leftColLim & burstLocCs<rightColLim);
    % row limits
    topRowLim = 205;
    bottomRowLim = -8.6444;
    burstLocCs = burstLocCs(burstLocRs>bottomRowLim & burstLocRs<topRowLim);
    burstLocRs = burstLocRs(burstLocRs>bottomRowLim & burstLocRs<topRowLim);
    % gray zone in between; if this occurs, keyboard and check out the data
    beginSecColInd =  -.5049;
    endFirstColInd = -.6422;
    if sum(burstLocCs < beginSecColInd & burstLocCs > endFirstColInd)> 0
        keyboard
    end
    
    burstNum = 0;% counter var to put in file name of saved output
    for iBurst = 1:2:length(burstLocCs) % count by 2 because two points per burst
        burstNum = burstNum + 1;
        firstCoordC = burstLocCs(iBurst);
        firstCoordR = burstLocRs(iBurst);
        secCoordC = burstLocCs(iBurst+1);
        secCoordR = burstLocRs(iBurst+1);
        
        %% check if pair of points is actually a pair
        if abs(firstCoordR - secCoordR) > 100
            keyboard
        end
        
        %% determine burst row
        lowerBounds_rows = [181.5619,154.7577,126.9227,101.9227,74.6031,47.7990,20.2216,-6.5825];
        if firstCoordR > lowerBounds_rows(1)
            row = 1;
        elseif firstCoordR > lowerBounds_rows(2)
            row = 2;
        elseif firstCoordR > lowerBounds_rows(3)
            row = 3;
        elseif firstCoordR > lowerBounds_rows(4)
            row = 4;
        elseif firstCoordR > lowerBounds_rows(5)
            row = 5;
        elseif firstCoordR > lowerBounds_rows(6)
            row = 6;
        elseif firstCoordR > lowerBounds_rows(7)
            row = 7;
        elseif firstCoordR > lowerBounds_rows(8)
            row = 8;
        else
            keyboard
        end
        
        %% determine burst col
        upperBoundFirstCol = -.65;
        if firstCoordC < upperBoundFirstCol
            col = 1;
        else
            col = 2;
        end
        allTrialsCol1Vals = [startTrial:2:endTrial-1];
        allTrialsCol2Vals = [startTrial+1:2:endTrial];
        allTrialsMatrix = [allTrialsCol1Vals',allTrialsCol2Vals'];
        
        trial = allTrialsMatrix(row,col);
        singleTrialRasterImageFileName = [multiTrialRasterPlotFile(1:beginTrialNumStartInd-1) num2str(trial) 't' num2str(trial) '.jpg'];
        
        %% load single trial image
        fullFileName = [singleTrialRasterImagesDir filesep singleTrialRasterImageFileName];
        if ~exist(fullFileName,'file') % create single trial image if it does not yet exist (exists for files with auto-detected bursts only thus far; manual burst may exist on trial without auto-detected burst)
            shuffleData = 0;
            % find stim number
            iStim = str2double(multiTrialRasterPlotFile((strfind(multiTrialRasterPlotFile,'Stim')+4):(strfind(multiTrialRasterPlotFile,'-trials')-1)));
            date = multiTrialRasterPlotFile((strfind(multiTrialRasterPlotFile,animal)+length(animal)+1):(strfind(multiTrialRasterPlotFile,animal)+length(animal)+1)+4);
            cond = multiTrialRasterPlotFile((strfind(multiTrialRasterPlotFile,animal)+length(animal)+7):(strfind(multiTrialRasterPlotFile,animal)+length(animal)+9));

            createSingleTrialFigToExtractWindow(date,cond,trial,trial,imageFormat,singleTrialRasterImagesDir,iStim,shuffleData)
        else
            % still get these vars for saving file later
            iStim = str2double(multiTrialRasterPlotFile((strfind(multiTrialRasterPlotFile,'Stim')+4):(strfind(multiTrialRasterPlotFile,'-trials')-1)));
            date = multiTrialRasterPlotFile((strfind(multiTrialRasterPlotFile,animal)+length(animal)+1):(strfind(multiTrialRasterPlotFile,animal)+length(animal)+1)+4);
            cond = multiTrialRasterPlotFile((strfind(multiTrialRasterPlotFile,animal)+length(animal)+7):(strfind(multiTrialRasterPlotFile,animal)+length(animal)+9));

        end
        rgbImage = imread(fullFileName);
        imshow(rgbImage);
        
        %% extract window around manually selected burst
        % load trial image and check size
        [rows columns numberOfColorBands] = size(rgbImage);
        if rows ~= 900 || columns ~= 1200
            crash
        end
        % average col cood's of selected burst to ascertain center of burst
        colCoord = (firstCoordC + secCoordC)/2;
        % convert colCoord to pixel column number
        pixelNum_firstTrialCol = 157;
        pixelNum_lastTrialCol = 1088;
        imageColRange = pixelNum_lastTrialCol - pixelNum_firstTrialCol;

        if col ==1
            colLowerLim = -2.1441;
            colUpperLim = -.6422;
            colRange = abs(colLowerLim - colUpperLim);
            colCoordAbs = abs(colLowerLim - colCoord);
            colFrac = colCoordAbs/colRange;
            pixelColRel = imageColRange*colFrac;
            pixelCenterCol = pixelColRel + pixelNum_firstTrialCol;
%             disp(['pixelCol:' num2str(pixelCol)])
        else
            colLowerLim = -.5049;
            colUpperLim = 1.0010;
            colRange = abs(colLowerLim - colUpperLim);
            colCoordAbs = abs(colLowerLim - colCoord);
            colFrac = colCoordAbs/colRange;
            pixelColRel = imageColRange*colFrac;
            pixelCenterCol = pixelColRel + pixelNum_firstTrialCol;
        end
        totalSecInTrial = trialXlimits(2) - trialXlimits(1);
        colsPerSec = columns/totalSecInTrial;
        colsPerWindow = round(colsPerSec*windowSize);
        
        leftIndex = round(pixelCenterCol - (colsPerWindow/2));
        rightIndex = round(pixelCenterCol + (colsPerWindow/2));
        
        if rightIndex - leftIndex ~= colsPerWindow
            keyboard
        end

        ca = rgbImage(:,leftIndex:rightIndex,:);
        rgbBlock = ca;
        jpgPlotFile = figure;
        imshow(rgbBlock); % Could call imshow(ca{r,c}) if you wanted to.
        saveFolder = [singleTrialRasterImagesDir filesep 'acceptedExamples' filesep 'manuallyDetected'];
        if ~exist(saveFolder,'dir')
            mkdir(saveFolder)
        end
        centeredWindowImage = ['RasterWindow-' animal '-' date '-' cond '-' num2str(iStim) '-trial' num2str(trial) 'burst' num2str(burstNum) imageFormat];
        saveas(jpgPlotFile,[saveFolder filesep centeredWindowImage])
        close all
    end
   
    
    
    
end