function plotSingleTrialRasters_CME_edits(exptType,exptDate,exptIndex,pars,isSpont,isLaminar)

if ~exist('isSpont','var')
    isSpont = 0;
end
if ~exist('isLaminar','var')
    isLaminar = 1;
end
[analysisComputer] = getAnalysisComputer();
nAnalysisComputers = length(analysisComputer);
tempDataPath = ['\Data\' exptType '\20' exptDate(1:2) '\' exptDate '-' exptIndex '\'];
analysisDirFound = 0;
iComputer = 0;
while ~analysisDirFound && iComputer<=nAnalysisComputers
    iComputer = iComputer+1;
    if exist([analysisComputer{iComputer} tempDataPath],'dir') == 7
        dataDir = [analysisComputer{iComputer} tempDataPath];
        analysisDirFound = 1;
    end
end
if ~analysisDirFound
    error('Data folder does not exist!');
end

exptInfo = [exptDate '-' exptIndex];
exptYear = ['20' exptDate(1:2)];

spikeDataFiles = dir([dataDir '*TrshldMUA*_noSnips.mat']);
if isempty(spikeDataFiles)
    spikeDataFiles = dir([dataDir '*TrshldMUA*.mat']);
end
% Sort spikeDataFiles by stim #
tempOFFIndex = zeros(length(spikeDataFiles),1);
for iFile = 1:length(spikeDataFiles)
    thisFile = spikeDataFiles(iFile).name;
    stimPos = strfind(thisFile,'Stim');
    underscorePos = strfind(thisFile,'_');
    thisUnderscore = underscorePos(find(underscorePos>stimPos(1),1,'first'));
    tempOFFIndex(iFile) = str2num(thisFile(stimPos(1)+4:thisUnderscore-1));
end
[~,sortIndex] = sort(tempOFFIndex);
spikeDataFiles = spikeDataFiles(sortIndex);

rawDataFiles = dir([dataDir '*_data*.mat']);
load([dataDir rawDataFiles(1).name],'dT');        
if pars.copyDataToThor
    copyfile([dataDir rawDataFiles(1).name],[pars.thorDir filesep 'rawData' filesep rawDataFiles(1).name])
end

trialFiles = dir([dataDir '*_trial*.mat']);
trialList_Concat = [];
for jFile = 1:max(size(trialFiles))
    load([dataDir trialFiles(jFile).name],'trialList');
    if pars.copyDataToThor
        copyfile([dataDir trialFiles(jFile).name],[pars.thorDir filesep 'rawData' filesep trialFiles(jFile).name])
    end
    if exist('trialList','var')
        display(['Processing ' trialFiles(jFile).name]);
        trialList_Concat = [trialList_Concat trialList];
        clear trialList
    end
end
trialList = trialList_Concat;
clear trialList_Concat

%Get stim parameter values
display('Connecting to dataBase...');
dbConn = dbConnect();
exptDate_dbForm = houseConvertDateTo_dbForm(exptDate);
exptIndex_dbForm = str2num(exptIndex);

display('Querying dataBase...');
masterResult = fetch(dbConn,['select exptid,animalID from masterexpt where exptDate=''' exptDate_dbForm ''' and exptIndex=' num2str(exptIndex_dbForm)]);
exptID = masterResult{1,1};
animalID = masterResult{1,2};
animalResult = fetch(dbConn,['select animalname,probe from animals where animalID=' num2str(animalID)]);
animalName = animalResult{1,1};
probeType = animalResult{1,2};
animalData = [animalName '-' exptDate '-' exptIndex];
if ~pars.doShuffle
    outFileName = [dataDir animalData 'MUA_UPstates.mat'];
else
    outFileName = [dataDir animalData 'MUA_UPstates_SHUFFLED.mat'];
end

switch probeType
    case {'NNC16','NNF16','NNHZ16','NNH16','NNHC16','NNC16old','NNH16old','NNA16','NNCM16','ATLAS1x16'}
        chanOrder = 1:16;
    case {'TDT2x8'} % this is for the zif style probe
        chanOrder = 1:16;
    %    chanOrder = [1,9,2,10,3,11,4,12,5,13,6,14,7,15,8,16];
    otherwise
        chanOrder = 1:16;
end

%Use function 'unique' to determine the number and index numbers of the
%distinct stimuli in trialList
tempStimID = [trialList(:).uniqueStimID]';
[distinctStimID,~,~] = unique(tempStimID);
nStim = length(distinctStimID);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%need to query db for stimulus parameter names and values based on StimIDs
%need these for two purposes: (1) to sort columns in plot, and (2) to
%determine t=0 (which we can set e.g. as the shortest latency auditory stim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('Querying database for Stimulus parameters');

%Get all of the parameter names from stimuli
stimFields = columns(dbConn,[],[],'stimuli');

%Build query to select all IDs in unique
stimQuery = 'select * from stimuli where';
for j=1:nStim
    tmpStr =  [' id=' num2str(distinctStimID(j))];
    if j~= nStim
        tmpStr = [tmpStr ' OR '];
    end
    stimQuery = strcat(stimQuery,tmpStr);
end

%Build a table  stimTable with id number and cell array of param names and values
%excluding the null values
stimValueResult = fetch(dbConn,stimQuery);
stimTable = [];

%Get list of stimulus parameters to use, and build a matrix of their values
paramList = {};
paramValues = [];
for par=2:size(stimValueResult,2)  %For each parameter in the stimuli table
    if sum(~isnan([stimValueResult{:,par}])) > 0 %If any stimulus has a non-NaN result
        paramList = [paramList stimFields(par)]; %Add to the parameter list
        paramValues = [paramValues , [stimValueResult{:,par}]']; %Add to the values table
    end
end

%%%%%%%%%%%%%
% Decide on sort order based on paramList
%%%%%%%%%%%%%

paramOrder = parameterSortOrder(paramList);

%%%%%%%%%%%%%
% Having sort order, use to sort by paramValues
%%%%%%%%%%%%%
[sortedStims,stimOrder] = sortrows(paramValues,paramOrder);
sortedStimIDs = distinctStimID(stimOrder);

%%%%%%%%%%%%%
% Find stim delays and LED dur
%%%%%%%%%%%%%
isDelay = false(1,length(paramList));
isAudDelay = false(1,length(paramList));
isLEDDur = false(1,length(paramList));
for p = 1:length(paramList)
    %Search for "del" in values to flag all fields that are delays
    if ~isempty(strfind(paramList{p},'del'))
        isDelay(p) = 1;
        if ~isempty(strfind(paramList{p},'spkr'))
            isAudDelay(p) = 1;
        end
    elseif ~isempty(strfind(paramList{p},'led')) && ~isempty(strfind(paramList{p},'dur'))
        isLEDDur(p) = 1;
    end
end
% Build an array of unique delays
allStimTimes = paramValues(:,isDelay);
allStimTimes = unique(allStimTimes(:));
allAudStimTimes = paramValues(:,isAudDelay);
allAudStimTimes = unique(allAudStimTimes(:));
shortest_tStim = min(allStimTimes(logical(allStimTimes>0)));
if ~isempty(allAudStimTimes)
    shortest_tAudStim = min(allAudStimTimes(logical(allAudStimTimes>0)));
else
    shortest_tAudStim = shortest_tStim;
end
allNonAudStimTimes = allStimTimes(~ismember(allStimTimes,allAudStimTimes));

nChans = pars.nChans;
if ~exist('pars.chansToUse','var')
    pars.chansToUse = 1:nChans;
end
if isLaminar
    chanShift = pars.actGranCh - pars.idealGranCh;
    actChPos = pars.idealChPos - chanShift*0.1;
    layerBounds = pars.layerBoundaries*(pars.nChans-1)*0.1; %in mm
    if chanShift>=0
        chanStart = chanShift+1;
        chanStop = nChans;
    else
        chanStart = 1;
        chanStop = nChans+chanShift;
    end
    nChans = chanStop-chanStart+1;

    %The following are the sets of channels that are averaged to yield data
    %by layer.
    chanVec = 1:pars.nChans;
    posVec = [0 layerBounds pars.nChans*0.1-0.05];
    for iLayer = 1:length(layerBounds)+1
        layer(iLayer).chans = chanVec(actChPos>posVec(iLayer) & actChPos<=posVec(iLayer+1))-chanStart+1;
    end
else
    chanStart = 1;
end

% Main program loop. Loop through spike data files for each stim
firstFile = 1;
if isempty(pars.filesToUse)
    pars.filesToUse = 1:length(spikeDataFiles);
end
for iFile = pars.filesToUse
    thisSpikeFile = spikeDataFiles(iFile).name;
    display(['Processing ' thisSpikeFile '...']);
    
    load([dataDir thisSpikeFile]);
    if pars.copyDataToThor
        copyfile([dataDir thisSpikeFile],[pars.thorDir filesep 'rawData' filesep thisSpikeFile])
    end
    
    underscorePos = strfind(thisSpikeFile,'_');
    stimIDPos = strfind(thisSpikeFile,'StimID');
    thisStimID = str2num(thisSpikeFile(stimIDPos+6:underscorePos(4)-1));
    stimNumber = thisSpikeFile(underscorePos(2)+1:underscorePos(3)-1);

    %For LED stimuli, will reject spikes that occur near LED artifact
    if sum(sortedStims(sortedStimIDs==thisStimID,isLEDDur))>0
        isLEDStim = 1;
    else
        isLEDStim = 0;
    end

    if firstFile
        spikesByChan_allStim(1:nChans) = struct('timesReOn_Evoked',[],'timesNmlz_Evoked',[],...
            'timesReOn_OffResp',[],'timesNmlz_OffResp',[],'timesReOn_Spont',[],'timesNmlz_Spont',[]);
        nTotSpikes_allStim = zeros(1,nChans);
        nEvSpikes_allStim = zeros(1,nChans);
        nBaseSpikes_allStim = zeros(1,nChans);
        nOnEv_allStim = 0;
        nOffEv_allStim = 0;
        nSpEv_allStim = 0;
        totalNumPts_allStim = 0;
        totalEvPts_allStim = 0;
        firstFile = 0;
    end
    tStim = shortest_tAudStim/1000;
    if exist('shortest_tAudDur','var') && ~isempty(shortest_tAudDur) %Then override user setting for pars.tAudDur
        if shortest_tAudDur>5 && shortest_tAudDur<=500 
            %<=500 is necessary because for vocaliz this parameter is set incorrectly in database
            %>5 so that clicks responses can also be processed
            %correctly
            pars.tAudDur = shortest_tAudDur/1000;
        end
    end

    %Find trials common to all channels for analysis
    if pars.nChans == 16
        trialsToUse = mintersect(trialInfo(1).trialNums,...
            trialInfo(2).trialNums,trialInfo(3).trialNums,...
            trialInfo(4).trialNums,trialInfo(5).trialNums,...
            trialInfo(6).trialNums,trialInfo(7).trialNums,...
            trialInfo(8).trialNums,trialInfo(9).trialNums,...
            trialInfo(10).trialNums,trialInfo(11).trialNums,...
            trialInfo(12).trialNums,trialInfo(13).trialNums,...
            trialInfo(14).trialNums,trialInfo(15).trialNums,...
            trialInfo(16).trialNums);
    elseif pars.nChans == 8
        trialsToUse = mintersect(trialInfo(1).trialNums,...
            trialInfo(2).trialNums,trialInfo(3).trialNums,...
            trialInfo(4).trialNums,trialInfo(5).trialNums,...
            trialInfo(6).trialNums,trialInfo(7).trialNums,...
            trialInfo(8).trialNums);
    else
        error('Non-standard #channels');
    end

    % Load spikes into convenient data structure
    nTrials = min([length(trialsToUse),pars.trialStop-pars.trialStart+1]);

    pars.nTrialsToPlot = min([nTrials,pars.nTrialsToPlot]);
    nPtsPerTrial = size(MUAPsth,2);
    totalNumPts = nPtsPerTrial*nTrials;
    if exist('spikes','var')
        clear spikes
    end
    spikes(1:nChans,1:nTrials) = struct('times',[]);

    for iPlot = 1:nTrials
        % spikeMatrix is just used to find instances of simultaneous
        % spikes on multiple channels, which are assumed artifactual.
        spikeMatrix = zeros(nChans,nPtsPerTrial);
        for iChan = 1:nChans
            dataOutChan = chanOrder(iChan);
            dataInChan = pars.chansToUse(iChan);
            thisTrial = find(trialInfo(dataInChan).trialNums==trialsToUse(iPlot+pars.trialStart-1),1,'first');
            if ~isempty(spikeData(dataInChan+chanStart-1).trial(thisTrial).times)
                if isLEDStim
                    spikesToKeep = false(1,length(spikeData(dataInChan+chanStart-1).trial(thisTrial).times));
                    for iStim = 1:length(allNonAudStimTimes)
                        spikesToKeep = spikesToKeep |...
                            spikeData(dataInChan+chanStart-1).trial(thisTrial).times<(allNonAudStimTimes(iStim)-2)/1000 |...
                            spikeData(dataInChan+chanStart-1).trial(thisTrial).times>(allNonAudStimTimes(iStim)+2)/1000;
                    end
                else
                    spikesToKeep = true(1,length(spikeData(dataInChan+chanStart-1).trial(thisTrial).times));
                end
                spikeMatrix(dataOutChan,round(spikeData(dataInChan+chanStart-1).trial(thisTrial).times(spikesToKeep)/dT)) = 1;
                spikes(dataOutChan,iPlot).times = spikeData(dataInChan+chanStart-1).trial(thisTrial).times(spikesToKeep);
            else
                spikes(dataOutChan,iPlot).times = [];
            end
        end
        %The following will be used later to check for spurious
        %noise appearing as simultaneous spikes on multiple
        %channels.
        testVec = find(sum(spikeMatrix,1)>2);
        for iTest =1:length(testVec)
            for iChan = 1:nChans
                tempTimes = round(spikes(iChan,iPlot).times/dT);
                testIndex = tempTimes==testVec(iTest);
                spikes(iChan,iPlot).times = tempTimes(~testIndex)*dT;
            end
        end
    end
    if pars.doShuffle
        for iChan = 1:nChans
            spikes(iChan,:) = spikes(iChan,randperm(nTrials));
        end
    end

    nSpkPts = round(nPtsPerTrial*dT*1000);
    time = (0:nSpkPts-1)/1000 - tStim;

    if pars.plotRastByTrial
        if ~isSpont
            nTrialsPerPlot = min([(pars.nRows*pars.nCols),nTrials]);
            for trialSet = 1:max([floor(nTrials/nTrialsPerPlot),1])
                trialStart = 1+(trialSet-1)*nTrialsPerPlot;
                trialStop = trialSet*nTrialsPerPlot;
                if ~pars.doShuffle
                    FigName = ['Raster plot by trial - ' animalName '-' exptInfo '-' ...
                        stimNumber '-trials' num2str(trialStart+pars.trialStart-1) 't' num2str(trialStop+pars.trialStart-1)];
                else
                    FigName = ['Raster plot by trial - ' animalName '-' exptInfo '-' ...
                        stimNumber '-trials' num2str(trialStart+pars.trialStart-1) 't'...
                        num2str(trialStop+pars.trialStart-1) '-SHUFFLED'];
                end
                figure('Units', 'normalized', 'Position', [0.01, 0.1, 0.98, 0.8],...
                    'Name', FigName);
                for iPlot = trialStart:trialStop               
                    subplot(pars.nRows,pars.nCols,iPlot-trialStart+1);
                    if iPlot == trialStart
                        if ~pars.doShuffle
                            title([animalName '-' exptInfo '-' stimNumber ': Trial #' num2str(iPlot+pars.trialStart-1)],'FontSize',12);
                        else
                            title([animalName '-' exptInfo '-' stimNumber ': Trial #' num2str(iPlot+pars.trialStart-1) '; SHUFFLED!'],'FontSize',12);
                        end
                    end
                    hold on
                    plotMatrix = zeros(nChans,nSpkPts);
                    for iChan = 1:nChans
                        spk1 = round(spikes(iChan,iPlot).times*1000)+1; %1 msec time resolution
                        train1 = zeros(nSpkPts,1);
                        train1(spk1(spk1<=nSpkPts)) = 1;
                        plotMatrix(nChans-iChan+1,:) = train1;
                    end
                    imagesc(time,1:nChans,plotMatrix);
                    colormap(gray);
                    if pars.plotStimIndicators
                        plot([0 0],pars.yPlotLim,'b','LineWidth',0.5);
                        plot([pars.tAudDur pars.tAudDur],pars.yPlotLim,'b','LineWidth',0.5);
                    end
                    xlim(pars.tPlotLim);
                    set(gca,'xTick',pars.tRastTicks);
                    set(gca,'yTick',pars.yRastByTrialTicks);
                    ylim(pars.yPlotLim);
                    if iPlot-trialStart+1 == (pars.nRows-1)*pars.nCols+1
                        set(gca,'xTickLabel',pars.tRastTicks,'FontSize',14);
                        set(gca,'yTickLabel',pars.yRastByTrialTickLabels,'FontSize',14);
                        xlabel('Time (sec)','FontSize',18);
                        ylabel('Chan #','FontSize',18);
                    else
                        set(gca,'xTickLabel',{});
                        set(gca,'yTickLabel',{});
                    end
                    sub_pos = get(gca,'position'); % get subplot axis position
                    set(gca,'position',sub_pos.*[1 1 1.2 1.3]) % stretch its width and height
                    set(gca,'tickDir','out')
                end
                if pars.savePlots
                    if pars.saveAllFigsTogetherOnThor
                        thorDir = pars.thorDir;
                        if isSpont
                            disp(['stimNum for spont...' num2str(stimNumber)])
                            saveDir = [thorDir filesep 'figs' filesep animalName filesep exptIndex filesep 'spont'];
                            if ~exist(saveDir,'dir')
                                mkdir(saveDir)
                            end
                        else
                            saveDir = [thorDir filesep 'figs' filesep animalName filesep exptIndex filesep stimNumber];
                            if ~exist(saveDir,'dir')
                                mkdir(saveDir)
                            end
                        end
                        
                        saveas(gcf,[saveDir filesep FigName '.fig']);
                    else
                        saveas(gcf,[dataDir FigName '.fig']);
                    end
                end
            end
        else
            tPlotDur = pars.tPlotLim(2)-pars.tPlotLim(1);
            tTrialDur = (nPtsPerTrial-1)*dT;
            tTotalDur = nTrials*tTrialDur;
            nPlotsTotal = floor(tTotalDur/tPlotDur);
            nTrialsPerPlot = min([(pars.nRows*pars.nCols),nPlotsTotal]);
            trialNum = 1;
            iPlotThisTrial = 0;
            for plotSet = 1:max([floor(nPlotsTotal/nTrialsPerPlot),1])
                plotStart = 1+(plotSet-1)*nTrialsPerPlot;
                plotStop = plotSet*nTrialsPerPlot;
                if ~pars.doShuffle
                    FigName = ['Raster plot by trial - ' animalName '-' exptInfo '-' ...
                        stimNumber '-trial' num2str(trialNum)];
                else
                    FigName = ['Raster plot by trial - ' animalName '-' exptInfo '-' ...
                        stimNumber '-trial' num2str(trialNum) '-SHUFFLED'];
                end
                figure('Units', 'normalized', 'Position', [0.01, 0.1, 0.98, 0.8],...
                    'Name', FigName);
                for iPlot = plotStart:plotStop
                    if iPlotThisTrial*tPlotDur>tTrialDur
                        trialNum = trialNum+1;
                        iPlotThisTrial = 0;
                    end
                    iPlotThisTrial = iPlotThisTrial+1;
                    if trialNum<=nTrials
                        subplot(pars.nRows,pars.nCols,iPlot-plotStart+1);
                        if iPlot == plotStart
                            if ~pars.doShuffle
                                title([animalName '-' exptInfo '-' stimNumber ': Trial #' num2str(trialNum)],'FontSize',12);
                            else
                                title([animalName '-' exptInfo '-' stimNumber ': Trial #' num2str(trialNum) '; SHUFFLED!'],'FontSize',12);
                            end
                        end
                        hold on
                        plotMatrix = zeros(nChans,nSpkPts);
                        for iChan = 1:nChans
                            spk1 = round(spikes(iChan,trialNum).times*1000)+1; %1 msec time resolution
                            train1 = zeros(nSpkPts,1);
                            train1(spk1(spk1<=nSpkPts)) = 1;
                            plotMatrix(nChans-iChan+1,:) = train1;
                        end
                        imagesc(time,1:nChans,plotMatrix);
                        colormap(gray);
                        if pars.plotStimIndicators
                            plot([0 0],pars.yPlotLim,'b','LineWidth',0.5);
                            plot([pars.tAudDur pars.tAudDur],pars.yPlotLim,'b','LineWidth',0.5);
                        end
                        xlimVals = pars.tPlotLim + (iPlotThisTrial-1)*tPlotDur;
                        xlim(xlimVals);
                        set(gca,'xTick',xlimVals);
                        set(gca,'yTick',pars.yRastByTrialTicks);
                        ylim(pars.yPlotLim);
                        set(gca,'xTickLabel',xlimVals,'FontSize',14);
                        if iPlot-plotStart+1 == (pars.nRows-1)*pars.nCols+1
                            set(gca,'yTickLabel',pars.yRastByTrialTickLabels,'FontSize',14);
                            xlabel('Time (sec)','FontSize',18);
                            ylabel('Chan #','FontSize',18);
                        else
                            set(gca,'yTickLabel',{});
                        end
                        sub_pos = get(gca,'position'); % get subplot axis position
                        set(gca,'position',sub_pos.*[1 1 1.2 1.1]) % stretch its width and height
                        set(gca,'tickDir','out')
                    end
                end
                if pars.savePlots
                    if pars.saveAllFigsTogetherOnThor
                        thorDir = pars.thorDir;
                        if isSpont
                            saveDir = [thorDir filesep 'figs' filesep animalName filesep exptIndex filesep 'spont'];
                            if ~exist(saveDir,'dir')
                                mkdir(saveDir)
                            end
                        else
                            saveDir = [thorDir filesep 'figs' filesep animalName filesep exptIndex filesep stimNumber];
                            if ~exist(saveDir,'dir')
                                mkdir(saveDir)
                            end
                        end
                        
                        saveas(gcf,[saveDir filesep FigName '.fig']);
                    else
                        saveas(gcf,[dataDir FigName '.fig']);
                    end
                end
            end
        end
    end
end

if pars.closeAll
    close all;
end

