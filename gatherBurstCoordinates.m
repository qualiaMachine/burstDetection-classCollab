function [burstLocC,burstLocR] = gatherBurstCoordinates(figDir,multiTrialFile)
rasterPlot = openfig([figDir filesep multiTrialFile]);
% set(rasterPlot,'units','normalized','outerposition',[0 0 1 1])
colormap default
[burstLocC, burstLocR] = getpts(gcf);
% set( gcf, 'crosshair' )
% getline(gcf)
%% remove last maker since this one will always be a junk marker
burstLocC = burstLocC(1:end-1); % -1.59 --> -.59
burstLocR = burstLocR(1:end-1);
close all;

end

